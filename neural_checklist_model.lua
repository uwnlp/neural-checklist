require 'cunn'
require 'nngraph'
require 'optim'
local model_utils = require('utils.utils')
local stringx = require('pl.stringx')
include 'layers/LookupTableWithNulls.lua'
include 'layers/LookupTableWithNullsNoUpdate.lua'
include 'layers/TrainedParamMult.lua'

local neural_checklist_model = {}
neural_checklist_model.__index = neural_checklist_model

--------------------------------------------------------
-- Creates a new neural checklist model.
--------------------------------------------------------
-- opt: the trainitem option flags
-- max_info: max length information for the data that the
--    model will be trained on (e.g., maximum number of 
--    agenda)
-- dict: recipe dictionary (used if there are pre-trained
--    embeddings to read in)
---------------------------------------------------------
function neural_checklist_model:new(opt, max_info, dict)
   local new_neural_checklist_model = {}
   setmetatable(new_neural_checklist_model, self)

   new_neural_checklist_model.opt = opt
   new_neural_checklist_model.max_info = max_info
   new_neural_checklist_model:build_models(dict)

   return new_neural_checklist_model
end


---------------------------------------------------------------------
-- Loads a neural checklist model for generation.
---------------------------------------------------------------------
--
-- opt: the generation option flags
-- info: the model file version information
-- epoch: the epoch number of the model beitem loaded
--
-- The three arguments help find the files (opt.checkpoint_dir is the
-- folder of models) and generate the filenames (using info and epoch).
----------------------------------------------------------------------
function neural_checklist_model:load_generation_model(opt, info, epoch)
   local loaded_model = {}
   setmetatable(loaded_model, self)
   local file_name = opt.checkpoint_dir .. 'neural_checklist_model.' .. info
   loaded_model.model = torch.load(file_name .. '.model.ep' .. epoch)
   loaded_model.embedder = torch.load(file_name .. '.embed.ep' .. epoch)
   loaded_model.outputter = torch.load(file_name .. '.outputter.ep' .. epoch)
   loaded_model.opt = torch.load(file_name .. '.opt')
   loaded_model.goal_encoders = {}
   for l=1,loaded_model.opt.num_layers do
      local goal_encoder = torch.load(file_name .. '.goal' .. l .. '.ep' .. epoch)
      table.insert(loaded_model.goal_encoders, goal_encoder)
   end
   loaded_model.max_info = torch.load(file_name .. '.max_info')
   loaded_model.gen_opt = opt
   loaded_model.num_steps = loaded_model.max_info.num_words

   return loaded_model
end

---------------------------------------------------
-- Comparison function for sorting the beam search.
---------------------------------------------------
-- (assumes log probabilities)
--------------------------------------------------
function beam_compare(a, b)
   return (a.prob * (1.0/a.len)) >(b.prob * (1.0 / b.len))
end

-------------------------------------------------------------------------
-- Generates text using beam search.
------------------------------------------------------------------------
-- ps: initialized tensors to represent the language model hidden state
--     (Will be zeroed out by this function to start a new generation.)
-- state: information about the current input (e.g., goal, agenda)
-- dict: Output vocabulary (Used to identify newline index, etc.)
-- vocab_size: Output vocabulary size 
-- item_weights (optional): Weights to multiply each of the item
--     embeddings by
--
--
-- Returns best generated string, best beam
--------------------------------------------------------------------------
function neural_checklist_model:get_prediction(ps, state, dict, vocab_size, item_weights)

   -- Zero out the hidden state tensors
   if self.opt.rnn_type == 'lstm' then
      for d=1,2*self.opt.num_layers do
         ps[d]:zero()
      end
   elseif self.opt.num_layers == 1 then
      ps:zero()
   else
      for d=1,self.opt.num_layers do
         ps[d]:zero()
      end
   end

   local checklist = torch.zeros(1, state.batch_len[3]):cuda()
 
   cutorch:synchronize()

   -- Compute item embeddings.
   local item_embedding = torch.zeros(1, state.batch_len[3], self.opt.rnn_size):contiguous():float():cuda()
   local item_tmp = self.embedder:forward(state.agenda)
   item_embedding:copy(item_tmp)
   -- If item weights are given, scale the item embeddings using the weights.
   if item_weights ~= nil then
      local stretched_weights = torch.expand(item_weights, state.batch_len[3], self.opt.rnn_size)
      item_embedding:cmul(stretched_weights)
   end

      
   -- Compute goal embeddings (the goal embedding and its projection to initialize the RNN).  
   local embedded_goal = nil
   local rnn_start = nil
   for l=1,self.opt.num_layers do
      rnn_start, embedded_goal = unpack(self.goal_encoders[l]:forward(state.goal))
      if self.opt.rnn_type == 'lstm' then
         ps[2*l]:copy(rnn_start)
      elseif self.opt.num_layers == 1 then
         ps:copy(rnn_start)
      else
         ps[l]:copy(rnn_start)
      end
   end

   -- Create sets of item tokens. This can be used to force certain tokens to be used in certain ways.
   -- (Not used for EMNLP.)
   local item_tokens = {}
   local item_tokens_by_idx = {}
   for i=1,state.batch_len[3] do
      item_tokens_by_idx[i] = {}
      for j=1,state.batch_len[4] do
         local item_token = nil
         item_token = state.agenda[1][i][j]
         if item_token ~= 0 then
            item_tokens[item_token] = true
            item_tokens_by_idx[i][item_token] = true
            local item_word = dict.index_to_symbol[item_token]
            if stringx.lfind(item_word, '_') then
               local split = stringx.split(item_word, '_')
               if #split == 2 then
                  local token = dict.symbol_to_index[split[1] ]
                  if token ~= nil then
                     item_tokens[token] = true
                     item_tokens_by_idx[i][token] = true
                  end
                  token = dict.symbol_to_index[split[2] ]
                  if token ~= nil then
                     item_tokens[token] = true
                     item_tokens_by_idx[i][token] = true
                  end
               end
            end
         end 
      end 
   end

   -- Initialize beam with start symbol
   local start_step = dict.symbol_to_index['<text>']
   local prev_word = torch.ones(1)
   prev_word[1] = start_step
   prev_word = prev_word:cuda()

   
   local zeros = torch.zeros(state.batch_len[3])
   local zeros2 = torch.zeros(3)
   local checklists = {}
   local ref_type = {}
   table.insert(checklists, zeros)
   table.insert(ref_type, zeros2)

   -- Create initial beam
   local default_beam = {prob = 0.0,
                         str = "",
                         first_word_str = "",
                         len = 1,
                         prev_word = prev_word,
                         items = item_embedding,
                         ps = ps,
                         checklist = checklist,
                         nsteps = 1,
                         checklists = checklists,
                         used_first = {},
                         used_items = {},
                         available_items = {},
                         is_item = is_item,
                         item_index_str = '',
                         last_was_item = false,
                         item_used_idx = 0,
                         num_used_items = 0,
                         used_item_idxs = {},
                         ref_type = ref_type}
   local beams = {[1] = default_beam}

   -- Run beam search.
   local prediction, beam = self:beam(state, item_embedding, dict, embedded_goal, beams, item_tokens, item_tokens_by_idx, use_true, self.gen_opt.use_first, true, false)
   return prediction, beam
end

function neural_checklist_model:beam(state, item_embedding, dict, goal, beams, item_tokens, item_tokens_by_idx, use_true, use_first, gen_full_recipe, turns)
   local end_step = dict.symbol_to_index['</text>']
   local start_step = dict.symbol_to_index['<text>']
   local line_break = dict.symbol_to_index['\n']
   local unk = dict.symbol_to_index['<unk>']

   local determiners = {}
   local the = dict.symbol_to_index['the']
   local a = dict.symbol_to_index['a']
   local an = dict.symbol_to_index['an']
   local all = dict.symbol_to_index['all']
   determiners[the] = true
   determiners[a] = true
   determiners[an] = true
   determiners[all] = true

   local ones_item = torch.ones(1, 1, state.batch_len[3]):contiguous():float():cuda()
   local start_len = beams[1].len

   local dummy_atten = torch.ones(1, state.batch_len[3]):contiguous():float():cuda()
   local max_used = 0

   local y = torch.ones(1)
   y[1] = start_step
   y = y:cuda()
   local dummy_reftype = torch.zeros(1, 3):contiguous():float():cuda()
   dummy_reftype[1][1] = 1.0

   local finished_beam = nil
   local finished_beam_prob = -1000000.0

   local cnt = 0
   while cnt < self.gen_opt.max_length do
      local new_beams = {}
      local used = false
      for i=1, #beams do
         -- If the beam isn't finished (i.e., end of recipe token, end of line token if turntaking), expand beam
         if (beams[i].prev_word[1] ~= end_step) and (gen_full_recipe or beams[i].prev_word[1] ~= line_break or beams[i].len == start_len + 1 or (beams[i].prev_word[1] == line_break and turns and beams[i].len == start_len)) then
            used = true
            local tmp = nil
            local output_hidden_state = nil
            local next_state = nil
            local next_checklist = torch.zeros(1, state.batch_len[3]):cuda()
            local ref_type = torch.zeros(3)
            if self.opt.evidence_type == 0 then
               tmp = self.model:forward({beams[i].prev_word, beams[i].ps, item_embedding, goal, beams[i].checklist, beams[i].checklist, dummy_reftype, dummy_atten, dummy_atten})
               output_hidden_state = tmp[2]
               next_state = tmp[3]
               ref_type:copy(tmp[6])
               next_checklist:copy(tmp[5])
            elseif self.opt.evidence_type == 1 then
               tmp = self.model:forward({beams[i].prev_word, beams[i].ps, item_embedding, goal, beams[i].checklist})
               output_hidden_state = tmp[1]
               next_state = tmp[2]
               ref_type:copy(tmp[4])
               next_checklist:copy(tmp[3])
            else
               print('Unknown evidence type ' .. self.opt.evidence_type .. '.')
               os.exit()
            end

            local output_tmp = self.outputter:forward({output_hidden_state, y})
            local fnodes = self.outputter.forwardnodes
            local word_vector = fnodes[#fnodes].data.mapindex[1].input[1][1]
            word_vector:div(self.gen_opt.temperature):exp()
            word_vector[unk] = 0.0000000001
            word_vector:div(torch.sum(word_vector))

            local probs, inds = word_vector:sort()
      
            local prev_checklist = beams[i].checklist
            local checklist_update = torch.zeros(1, state.batch_len[3]):cuda()
            checklist_update:copy(next_checklist)
            checklist_update:add(-1, prev_checklist)

            local curr_item_idx = 1
            local curr_prob = checklist_update[1][1]
            for w = 2, state.batch_len[3] do
               if checklist_update[1][w] > curr_prob then
                  curr_prob = checklist_update[1][w]
                  curr_item_idx = w
               end
            end

            local usable_items = {}
            for w = 1,checklist_update:size(2) do
               if checklist_update[1][w] > 0.5 then
                  for token,_ in pairs(item_tokens_by_idx[w]) do
                     usable_items[token] = true
                  end
               end
            end

            local curr_idx = 1
            local okay = true

            local num_expansions_to_use = self.gen_opt.beam_size
            if use_true then
               num_expansions_to_use = 1
            end

            for j=1, num_expansions_to_use do
               local p = nil
               local ind = nil
               if use_true then
                  for v=1,self.max_info.vocab_size do
                     ind = inds[-1*v]
                     if ind == state.text[beams[i].len+1][1] then
                        p = probs[-1*v]
                        break
                     end
                  end
                  if p == nil then
                     print(dict.index_to_symbol[state.text[beams[i].len+1][1] ])
                     print('cant find true word')
                     os.exit()
                  end
               else
                  if curr_idx > probs:size(1) then
                     okay = false
                     break
                  end
                  if self.gen_opt.use_sampling then
                     local index = torch.multinomial(probs:float(), 1):resize(1):float()
                     p = probs[index[1]]
                     ind = inds[index[1]]
                  else
                     p = probs[-1* curr_idx]
                     ind = inds[-1*curr_idx]
                  end
                  if not okay then
                     break
                  end
                  if self.gen_opt.force_different_first_tokens and (beams[i].prev_word[1] == line_break or beams[i].prev_word[1] == start_step) then
                     local ind_to_check = ind
                     local item_word = dict.index_to_symbol[ind]
                     if stringx.lfind(item_word, '_') then
                        local split = stringx.split(item_word, '_')
                        if #split == 2 then
                           local token = dict.symbol_to_index[split[1] ]
                           ind_to_check = token
                        end
                     end 

                     while beams[i].used_first[ind_to_check] == true do
                        curr_idx = curr_idx + 1
                        if curr_idx > probs:size(1) then
                           okay = false
                           break
                        end
                        p = probs[-1* curr_idx]
                        ind = inds[-1*curr_idx]
                        local item_word = dict.index_to_symbol[ind]
                        if stringx.lfind(item_word, '_') then
                           local split = stringx.split(item_word, '_')
                           if #split == 2 then
                              local token = dict.symbol_to_index[split[1] ]
                              if token ~= nil then
                                 ind_to_check = token
                              end
                           end
                        end 
                     end
                     if not okay then
                        break
                     end
                  end
               end
               if not okay then
                  break
               end

               curr_idx = curr_idx + 1

               local next_used_first = {}
               for used,_ in pairs(beams[i].used_first) do
                  next_used_first[used] = true
               end

               if beams[i].prev_word[1] == line_break or beams[i].prev_word[1] == start_step then
                  local ind_to_check = ind
                  local item_word = dict.index_to_symbol[ind]
                  if stringx.lfind(item_word, '_') then
                     local split = stringx.split(item_word, '_')
                     if #split == 2 then
                        local token = dict.symbol_to_index[split[1] ]
                        if token ~= nil then
                           ind_to_check = token
                        end
                     end
                  end 
                  next_used_first[ind_to_check] = true
               end

               local next_used_items = {}
               for used,_ in pairs(beams[i].used_items) do
                  next_used_items[used] = true
               end

               if ref_type[3] > (0.5) and determiners[ind] ~= true then
                  next_used_items[ind] = true
               end

               local next_item_index_str = beams[i].item_index_str
               local next_num_used_items = beams[i].num_used_items
               local next_used_item_idxs = {}
               for used,_ in pairs(beams[i].used_item_idxs) do
                  next_used_item_idxs[used] = true
               end
               local next_item_used_idx = beams[i].item_used_idx
               if beams[i].used_item_idxs[curr_item_idx] == nil and ref_type[3] > (0.5) then
                  next_item_index_str = next_item_index_str .. ' ' .. tostring(curr_item_idx)
                  next_used_item_idxs[curr_item_idx] = true 
                  next_item_used_idx = next_item_used_idx + math.pow(2,curr_item_idx - 1)
                  next_num_used_items = next_num_used_items + 1
               end

               local next_word = torch.ones(1)
               next_word[1] = ind
               next_word = next_word:cuda()

               local next_str = beams[i].str
               if next_str == '' then
                  next_str = dict.index_to_symbol[ind]
               else
                  next_str = next_str .. ' ' .. dict.index_to_symbol[ind]
               end
               local next_len = beams[i].len + 1

               local next_ps = nil
               if self.opt.rnn_type == 'lstm' then
                  next_ps = {}
                  for d = 1, self.opt.num_layers do
                     next_ps[(2*d)-1] = torch.zeros(1, self.opt.rnn_size):cuda()
                     next_ps[(2*d)] = torch.zeros(1, self.opt.rnn_size):cuda()
                  end
                  model_utils.copy_table(next_ps, next_state)
               elseif self.opt.num_layers == 1 then
                  next_ps = torch.zeros(1, self.opt.rnn_size):cuda()
                  next_ps:copy(next_state)
               else
                  next_ps = {}
                  for d = 1, self.opt.num_layers do
                     next_ps[d] = torch.zeros(1, self.opt.rnn_size):cuda()
                  end
                  model_utils.copy_table(next_ps, next_state)
               end

               local next_checklist_duplicate = torch.zeros(1, state.batch_len[3]):cuda()
               next_checklist_duplicate:copy(next_checklist)
               local new_checklists = {}
               for _,probs in ipairs(beams[i].checklists) do
                  table.insert(new_checklists, probs)
               end
               table.insert(new_checklists, next_checklist_duplicate)

               local new_ref_type = {}
               for _,probs in ipairs(beams[i].ref_type) do
                  table.insert(new_ref_type, probs)
               end
               table.insert(new_ref_type, ref_type)

               local next_first_word_str = beams[i].first_word_str
               if (beams[i].prev_word[1] == start_step) then
                  next_first_word_str = beams[i].first_word_str .. ' ' .. ind
               end

               local next_beam = {prob = (beams[i].prob + math.log(p)),
                         str = next_str,
                         first_word_str = next_first_word_str,
                         len = next_len,
                         prev_word = next_word,
                         nsteps = beams[i].nsteps,
                         ps = next_ps,
                         checklist = next_checklist,
                         ref_type = new_ref_type,
                         item_index_str = next_item_index_str,
                         used_item_idxs = next_used_item_idxs,
                         num_used_items = next_num_used_items,
                         item_used_idx = next_item_used_idx,
                         last_was_item = (ref_type[3] > (0.5)),
                         used_first = next_used_first,
                         used_items = next_used_items,
                         checklists = new_checklists}

               if use_true and gen_full_recipe and next_beam.prev_word[1] == end_step then
                  return next_beam.str, next_beam
               end
               if use_true and (not gen_full_recipe) and (state.text[beams[i].len + 2][1] == end_step or state.text[beams[i].len + 2][1] == line_break) then
                  return next_beam.str, next_beam
               end
               table.insert(new_beams, next_beam)
   
               if beams[i].last_was_item and next_beam.last_was_item then
                  next_beam.prob = next_beam.prob - 10000
               end
               if beams[i].prev_word[1] == line_break then
                  next_beam.nsteps = next_beam.nsteps + 1
               end
               table.insert(new_beams, next_beam)
            end
         else
            table.insert(new_beams, beams[i])
         end
      end
      table.sort(new_beams, beam_compare)
      beams = {}
      for k=self.gen_opt.beam_size + 1, #new_beams do
         new_beams[k] = nil
      end
      beams = new_beams

      for k=1,#new_beams do
         local next_beam = new_beams[k]
         if next_beam.prev_word[1] == end_step and beams[k].num_used_items >= max_used then
            max_used = beams[k].num_used_items
            local end_err = (beams[k].prob *(1.0/beams[k].len))
            if end_err > finished_beam_prob then
               finished_beam = beams[k]
               finished_beam_prob = end_err
            end
         end
      end
      cnt = cnt + 1
   end

   if gen_full_recipe then
      for k=1,#beams do
         if beams[k].prev_word[1] == end_step then
            local num_end = math.abs(state.batch_len[3] - beams[k].checklist:sum())
            local num_used = (beams[k].prob * (1.0/beams[k].len)) - (num_end)
            if beams[k].prev_word[1] == end_step and beams[k].num_used_items >= max_used then
               if beams[k].num_used_items >= max_used then
                  max_used = beams[k].num_used_items
                  local end_err = (beams[k].prob *(1.0/beams[k].len))
                  if end_err > finished_beam_prob then
                     finished_beam = beams[k]
                     finished_beam_prob = end_err
                  end
               end
            end
         end
      end
   end
   if finished_beam == nil then
      finished_beam = beams[1]
   end
   local combined_atten = torch.zeros(state.batch_len[3], #(finished_beam.checklists))
   for i,atten in ipairs(finished_beam.checklists) do
      combined_atten:narrow(2,i,1):copy(atten)
   end
   print(combined_atten)

   local combined_ref_type = torch.zeros(3, #(finished_beam.ref_type))
   for i,atten in ipairs(finished_beam.ref_type) do
      combined_ref_type:narrow(2,i,1):copy(atten)
   end
   print(combined_ref_type)
   return finished_beam.str, finished_beam
end



function adapted_gru(opt, prev_word, prev_h, agenda, prev_checklist, goal)
   local rev_atten = nn.AddConstant(1)(nn.MulConstant(-1)(nn.HardTanh()(prev_checklist)))
   local proj_items = nil
   if opt.sumnotmean then
      proj_items = nn.Sum(3)(TrainedParamMult(opt.rnn_size, opt.rnn_size)(nn.Transpose({2,3})(nn.MM(true, false)({nn.View(-1, 1):setNumInputDims(1)(rev_atten), agenda}))))
   else
      proj_items = nn.Mean(3)(TrainedParamMult(opt.rnn_size, opt.rnn_size)(nn.Transpose({2,3})(nn.MM(true, false)({nn.View(-1, 1):setNumInputDims(1)(rev_atten), agenda}))))
   end

   local proj_goal = nn.Linear(opt.rnn_size, opt.rnn_size)(goal)

   function new_input_sum_full(opt)
      local i2h            = nn.Linear(opt.rnn_size, opt.rnn_size)
      local h2h            = nn.Linear(opt.rnn_size, opt.rnn_size)
      return nn.CAddTable()({i2h(prev_word), h2h(prev_h)})
   end

   local update_gate = nn.Sigmoid()(new_input_sum_full(opt)):annotate{name='update_gate'}
   local reset_gate = nn.Sigmoid()(new_input_sum_full(opt)):annotate{name='reset_gate'}
   local item_gate = nn.Sigmoid()(new_input_sum_full(opt)):annotate{name='item_gate'}
   local goal_gate = nn.Sigmoid()(new_input_sum_full(opt)):annotate{name='goal_gate'}
   local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
   local gated_item = nn.CMulTable()({item_gate, proj_items})
   local gated_goal = nn.CMulTable()({goal_gate, proj_goal})
   local p2 = nn.Linear(opt.rnn_size, opt.rnn_size)(gated_hidden)
   local p3 = nn.Linear(opt.rnn_size, opt.rnn_size)(gated_item)
   local p1 = nn.Linear(opt.rnn_size, opt.rnn_size)(prev_word)
   local p4 = nn.Linear(opt.rnn_size, opt.rnn_size)(gated_goal)
   local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2,p3,p4}))
   local zh = nn.CMulTable()({update_gate, hidden_candidate})
   local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
   local next_h = nn.CAddTable()({zh, zhm1})

   return next_h
end

----------------------------------------------------------------------------------------
-- The GRU (above) was created to include as input the goal and available agenda items.
-- The LSTM was not evaluated. This function is a potential way to include the goal and
-- agenda, but there might be better ways. ...for someone else to investigate. :)
----------------------------------------------------------------------------------------
function adapted_lstm(opt, prev_word, prev_c, prev_h, agenda, prev_checklist, goal)
   local rev_atten = nn.AddConstant(1)(nn.MulConstant(-1)(nn.HardTanh()(prev_checklist)))
   local proj_items = nil
   if opt.sumnotmean then
      proj_items = nn.Sum(3)(TrainedParamMult(opt.rnn_size, opt.rnn_size)(nn.Transpose({2,3})(nn.MM(true, false)({nn.View(-1, 1):setNumInputDims(1)(rev_atten), agenda}))))
   else
      proj_items = nn.Mean(3)(TrainedParamMult(opt.rnn_size, opt.rnn_size)(nn.Transpose({2,3})(nn.MM(true, false)({nn.View(-1, 1):setNumInputDims(1)(rev_atten), agenda}))))
   end

   local proj_goal = nn.Linear(opt.rnn_size, opt.rnn_size)(goal)

   function new_input_sum(opt)
      local i2h = nn.Linear(opt.rnn_size, opt.rnn_size)
      local h2h = nn.Linear(opt.rnn_size, opt.rnn_size)
      return nn.CAddTable()({i2h(prev_word), h2h(prev_h)})
   end

   local in_gate = nn.Sigmoid()(new_input_sum(opt)):annotate{name = 'in_gate'}
   local forget_gate = nn.Sigmoid()(new_input_sum(opt)):annotate{name = 'forget_gate'}
   local item_gate = nn.Sigmoid()(new_input_sum(opt)):annotate{name = 'item_gate'}
   local goal_gate = nn.Sigmoid()(new_input_sum(opt)):annotate{name = 'goal_gate'}
   local in_gate2 = nn.Tanh()(new_input_sum(opt)):annotate{name = 'in_gate2'}

   local next_c = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({item_gate, proj_items}),
      nn.CMulTable()({goal_gate, proj_goal}),
      nn.CMulTable()({in_gate, in_gate2})
   })

   local out_gate = nn.Sigmoid()(new_input_sum(opt)):annotate{name = 'out_gate'}
   local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

   return next_h, next_c
end


-------------------------------------------------------------------------------------------------------
-- Creates the portions of the neural checklist model relatitem to the ref-type()
--  classifier and the attention models with evidence about the true ref-type() and attention.
-------------------------------------------------------------------------------------------------------
-- next_h: Hidden state computed by the language model
-- prev_checklist_with_evid: Previous checklist tensor using tokens, true attentions, and true ref-type() as evidence
-- prev_checklist_no_evid: Previous checklist tensor using only the tokens as evidence
-- agenda: item embedding matrix
-- true_reftype: true ref-type() values
-- true_new_item_atten: True values for the new item attention distribution
-- true_used_item_atten: True values for the used item attention distribution
-------------------------------------------------------------------------------------------------------
function attention_switch(self, next_h, prev_checklist_with_evid, prev_checklist_no_evid, agenda, true_reftype, true_new_item_atten, true_used_item_atten)
   -- Generate available items by multiplyitem by (1.0 - checklist)
   local rev_atten = nn.AddConstant(1)(nn.MulConstant(-1)(nn.HardTanh()(prev_checklist_with_evid))):annotate{name='attention switch old'}
   local rep_rev_atten = nn.Replicate(self.opt.rnn_size, 2,1)(rev_atten)
   local new_items = nn.CMulTable()({rep_rev_atten, agenda})

   -- Compute attention model over available agenda
   local ht2h = nn.Linear(self.opt.rnn_size, self.opt.rnn_size)(next_h)
   local viewed_prev = nn.Reshape(self.opt.rnn_size, 1, true)(ht2h)
   local dot_context = nn.Sum(3)((nn.MM(false, false, 'dot context')({new_items, viewed_prev})))
   local new_item_attention = nn.SoftMax()(nn.MulConstant(self.opt.attention_temperature)(dot_context))

   local viewed_true_new_item_atten = nn.View(-1, 1):setNumInputDims(1)(true_new_item_atten)
   local viewed_true_used_item_atten = nn.View(-1, 1):setNumInputDims(1)(true_used_item_atten)

   -- Compute available item attention embeddings using predicted and true attention information.
   local attention = nn.View(-1, 1):setNumInputDims(1)(new_item_attention)
   local new_item_enc = nn.Mean(3)(nn.MM(true, false)({agenda, attention}))
   local new_item_enc_with_evid = nn.Mean(3)(nn.MM(true, false)({agenda, viewed_true_new_item_atten}))

   -- Create used item embeddings.
   local rep_atten = nn.Replicate(self.opt.rnn_size, 2,1)(nn.HardTanh()(prev_checklist_with_evid))
   local used_item_embeddings = nn.CMulTable()({rep_atten, agenda})

   -- Compute attention model over used agenda.
   local used_item_dot_context = nn.Sum(3)((nn.MM(false, false, 'comp dot')({used_item_embeddings, viewed_prev})))
   local used_item_attention = nn.SoftMax()(nn.MulConstant(self.opt.attention_temperature)(used_item_dot_context))
   local viewed_used_item_atten = nn.View(-1, 1):setNumInputDims(1)(used_item_attention)

   -- Compute used item attention embeddings using predicted and true attention information.
   local used_item_enc = nn.Mean(3)(nn.MM(true, false)({agenda, nn.View(-1, 1):setNumInputDims(1)(used_item_attention)}))
   local used_item_enc_with_evid = nn.Mean(3)(nn.MM(true, false)({agenda, viewed_true_used_item_atten}))


   -- Compute ref-type() values.
   local proj_next_h = nn.Linear(self.opt.rnn_size, 3)
   local reftype = nn.SoftMax()(nn.MulConstant(self.opt.switch_temperature)(proj_next_h(next_h)))
   local nonitem_wins_one = nn.Select(2,1)(reftype)
   local used_item_wins_one = nn.Select(2,2)(reftype)
   local new_item_wins_one = nn.Select(2,3)(reftype)

   -- Expand the ref-type() values to be as large as the embeddings.
   local lm_wins = nn.Replicate(self.opt.rnn_size, 2)(nonitem_wins_one)
   local new_item_wins = nn.Replicate(self.opt.rnn_size, 2)(new_item_wins_one)
   local used_item_wins = nn.Replicate(self.opt.rnn_size, 2)(used_item_wins_one)

   -- Extract true ref-type() values.
   local true_nonitem_wins_one = nn.Select(2,1)(true_reftype)
   local true_used_item_wins_one = nn.Select(2,2)(true_reftype)
   local true_new_item_wins_one = nn.Select(2,3)(true_reftype)
   local true_new_item_wins = nn.Replicate(self.opt.rnn_size, 2)(true_new_item_wins_one)
   local true_used_item_wins = nn.Replicate(self.opt.rnn_size, 2)(true_used_item_wins_one)
   local true_nonitem_wins = nn.Replicate(self.opt.rnn_size, 2)(true_nonitem_wins_one)

   -- Compute output embeddings using predicted and true ref-type() values.
   local output_hidden_state_no_evid = nil
   local output_hidden_state_w_evid = nil
   if not self.opt.lm_only then
      output_hidden_state_no_evid = nn.CAddTable()({nn.CMulTable()({lm_wins, ht2h}), nn.CMulTable()({used_item_wins, used_item_enc}), nn.CMulTable()({new_item_wins, new_item_enc})})
      output_hidden_state_w_evid = nn.CAddTable()({nn.CMulTable()({true_nonitem_wins, ht2h}), nn.CMulTable()({true_used_item_wins, used_item_enc_with_evid}), nn.CMulTable()({true_new_item_wins, new_item_enc_with_evid})})
   end

   -- Scale attentions by the predicted or true ref-type() information.
   local new_item_atten_no_evid = nn.Sum(3)(nn.MM(false, false)({attention, nn.View(-1, 1, 1):setNumInputDims(1)(new_item_wins_one)}))
   local new_item_atten_with_reftype_evid = nn.Sum(3)(nn.MM(false, false)({attention, nn.View(-1, 1, 1):setNumInputDims(1)(true_new_item_wins_one)}))
   local new_item_atten_with_all_evid = nn.Sum(3)(nn.MM(false, false)({viewed_true_new_item_atten, nn.View(-1, 1, 1):setNumInputDims(1)(true_new_item_wins_one)}))
   local used_item_atten_with_reftype_evid = nn.Sum(3)(nn.MM(false, false)({viewed_used_item_atten, nn.View(-1, 1, 1):setNumInputDims(1)(true_used_item_wins_one)}))

   -- Evaluate predicted attention probabilities to true probabilities.
   local new_item_atten_err = nn.MSECriterion()({new_item_atten_with_reftype_evid, true_new_item_atten})
   local used_item_atten_err = nn.MSECriterion()({used_item_atten_with_reftype_evid, true_used_item_atten})

   -- Evaluate predicted ref-type() values to true values, scaled by opt.switchmul
   local mul_reftype = nn.MulConstant(self.opt.switchmul)(reftype)
   local true_mul_reftype = nn.MulConstant(self.opt.switchmul)(true_reftype)
   local reftype_err = nn.MSECriterion()({mul_reftype, true_mul_reftype})

   -- Update checklist (both using the predicted information and the evidence).
   local next_checklist_no_evid = nn.CAddTable()({new_item_atten_no_evid, prev_checklist_no_evid})
   local next_checklist_with_evid = nn.CAddTable()({new_item_atten_with_all_evid, prev_checklist_with_evid})

   if self.opt.lm_only then -- for ablation
      return next_h, next_h, next_checklist_with_evid, next_checklist_no_evid, reftype, reftype_err, new_item_atten_err, used_item_atten_err 
   else
      return output_hidden_state_w_evid, output_hidden_state_no_evid, next_checklist_with_evid, next_checklist_no_evid, reftype, reftype_err, new_item_atten_err, used_item_atten_err
   end
end

-------------------------------------------------------------------------------------------------------
-- Creates the portions of the neural checklist model relatitem to the ref-type()
--  classifier and the attention models without evidence.
-------------------------------------------------------------------------------------------------------
-- next_h: Hidden state computed by the language model
-- prev_checklist_no_evid: Previous checklist tensor using only the tokens as evidence
-- agenda: item embedding matrix
-------------------------------------------------------------------------------------------------------
function attention_switch_no_evidence(self, next_h, prev_checklist_no_evid, agenda)
   -- Generate available items by multiplyitem by (1.0 - checklist)
   local rev_atten = nn.AddConstant(1)(nn.MulConstant(-1)(nn.HardTanh()(prev_checklist_no_evid))):annotate{name='attention switch old'}
   local rep_rev_atten = nn.Replicate(self.opt.rnn_size, 2,1)(rev_atten)
   local new_items = nn.CMulTable()({rep_rev_atten, agenda})

   -- Compute attention model over available agenda
   local ht2h = nn.Linear(self.opt.rnn_size, self.opt.rnn_size)(next_h)
   local viewed_prev = nn.Reshape(self.opt.rnn_size, 1, true)(ht2h)
   local dot_context = nn.Sum(3)((nn.MM(false, false, 'dot context')({new_items, viewed_prev})))
   local new_item_attention = nn.SoftMax()(nn.MulConstant(self.opt.attention_temperature)(dot_context))

   -- Compute available item attention embeddings using predicted attention information.
   local attention = nil
   local attention = nn.View(-1, 1):setNumInputDims(1)(new_item_attention)
   local new_item_enc = nn.Mean(3)(nn.MM(true, false)({agenda, attention}))

   -- Create used item embeddings.
   local rep_atten = nn.Replicate(self.opt.rnn_size, 2,1)(nn.HardTanh()(prev_checklist_no_evid))
   local used_item_embeddings = nn.CMulTable()({rep_atten, agenda})

   -- Compute attention model over used agenda.
   local used_item_dot_context = nn.Sum(3)((nn.MM(false, false, 'comp dot')({used_item_embeddings, viewed_prev})))
   local used_item_attention = nn.SoftMax()(nn.MulConstant(self.opt.attention_temperature)(used_item_dot_context))

   -- Compute used item attention embeddings using predicted and attention information.
   local used_item_enc = nn.Mean(3)(nn.MM(true, false, 'comp enc')({agenda, nn.View(-1, 1):setNumInputDims(1)(used_item_attention)}))


   -- Compute ref-type() values.
   local proj_next_h = nn.Linear(self.opt.rnn_size, 3)
   local reftype = nn.SoftMax()(nn.MulConstant(self.opt.switch_temperature)(proj_next_h(next_h)))
   local nonitem_wins_one = nn.Select(2,1)(reftype)
   local used_item_wins_one = nn.Select(2,2)(reftype)
   local new_item_wins_one = nn.Select(2,3)(reftype)

   -- Expand the ref-type() values to be as large as the embeddings.
   local lm_wins = nn.Replicate(self.opt.rnn_size, 2)(nonitem_wins_one)
   local new_item_wins = nn.Replicate(self.opt.rnn_size, 2)(new_item_wins_one)
   local used_item_wins = nn.Replicate(self.opt.rnn_size, 2)(used_item_wins_one)

   -- Compute output embeddings using predicted ref-type() values.
   local output_hidden_state = nn.CAddTable()({nn.CMulTable()({lm_wins, ht2h}), nn.CMulTable()({used_item_wins, used_item_enc}), nn.CMulTable()({new_item_wins, new_item_enc})})

   -- Scale attention by the predicted ref-type() information.
   local new_item_atten = nn.Sum(3)(nn.MM(false, false)({attention, nn.View(-1, 1, 1):setNumInputDims(1)(new_item_wins_one)}))

   -- Update checklist (both using the predicted information and the evidence).
   local next_checklist_no_evid = nn.CAddTable()({new_item_atten, prev_checklist_no_evid})

   if self.opt.lm_only then -- for ablation
      return reftype, next_checklist_no_evid, next_h
   else
      return reftype, next_checklist_no_evid, output_hidden_state
   end
end

----------------------------------------------
-- Builds the neural checklist model.
----------------------------------------------
function neural_checklist_model:build_model()
   local prev_word = nn.Identity()()
   local prev_word_state = nn.Identity()()
   local agenda = nn.Identity()()
   local goal = nn.Identity()()
   local prev_checklist_no_evid = nn.Identity()()
   local prev_checklist_with_evid = nil
   local true_reftype = nil
   local true_new_item_atten = nil
   local true_used_item_atten = nil
   local prev_checklist_for_lm = prev_checklist_no_evid
   if self.opt.evidence_type == 0 then
      prev_checklist_with_evid = nn.Identity()()
      true_reftype = nn.Identity()()
      true_new_item_atten = nn.Identity()()
      true_used_item_atten = nn.Identity()()
      prev_checklist_for_lm = prev_checklist_with_evid
   end

   
   -- Lookup previous word.
   local word_lookup = nil
   if self.opt.embeddings ~= '' then
      word_lookup = LookupTableWithNullsNoUpdate(self.max_info.vocab_size, self.opt.rnn_size)
   else
      word_lookup = LookupTableWithNulls(self.max_info.vocab_size, self.opt.rnn_size)
   end
   table.insert(self.lookups, word_lookup)
   local prev_word_input = {[0] = word_lookup(prev_word)}

   -- Split the previous hidden state if using extra layers and/or LSTMs.
   local next_output = nil
   local next_items = nil
   local next_state = {}
   local splitted = nil
   if self.opt.rnn_type == 'lstm' then
      splitted = {prev_word_state:split(2 * self.opt.num_layers)}
   elseif self.opt.num_layers == 1 then
      splitted = prev_word_state
   else
      splitted = {prev_word_state:split(self.opt.num_layers)}
   end

   self.dropouts = {}
   for layer_idx = 1, self.opt.num_layers do  -- for each layer...
      local prev_c = nil
      local prev_h = nil
      if self.opt.rnn_type == 'lstm' then
         prev_c = splitted[2 * layer_idx - 1]
         prev_h = splitted[2 * layer_idx]
      elseif self.opt.num_layers == 1 then
         prev_h = splitted
      else
         prev_h = splitted[layer_idx]
      end
   
      local dropper = nn.Dropout(self.opt.dropout)
      table.insert(self.dropouts, dropper)
      local dropped = dropper(prev_word_input[layer_idx - 1])

      -- Generate next hidden state using language model
      if self.opt.rnn_type == 'lstm' then
         local next_h, next_c = lstm(self.opt, dropped, prev_c, prev_h, agenda, prev_checklist_for_lm, goal)
         table.insert(next_state, next_c)
         table.insert(next_state, next_h)
         prev_word_input[layer_idx] = next_h
      elseif self.opt.rnn_type == 'rnn' then
         print('todo rnn')
         os.exit(1)
      else
         local next_h = adapted_gru(self.opt, dropped, prev_h, agenda, prev_checklist_for_lm, goal)
         table.insert(next_state, next_h)
         prev_word_input[layer_idx] = next_h
      end
   end

   local next_h_out = prev_word_input[self.opt.num_layers]

   local reftype = nil
   local reftype_err = nil
   local output_hidden_state_w_evid = nil
   local output_hidden_state_no_evid = nil
   local next_checklist_with_evid = nil
   local next_checklist_no_evid = nil
   local new_item_atten_err = nil
   local used_item_atten_err = nil

   local module = nil

   -- Compute output hidden state and update checklist.
   if self.opt.evidence_type == 0 then -- have ref-type() and attention evidence
      output_hidden_state_w_evid, output_hidden_state_no_evid, next_checklist_with_evid, next_checklist_no_evid, reftype, reftype_err, new_item_atten_err, used_item_atten_err = attention_switch(self, next_h_out, prev_checklist_with_evid, prev_checklist_no_evid, agenda, true_reftype, true_new_item_atten, true_used_item_atten)

   -- Create gModule
      module = nn.gModule({prev_word, prev_word_state, agenda, goal, prev_checklist_with_evid, prev_checklist_no_evid, true_reftype, true_new_item_atten, true_used_item_atten}, {output_hidden_state_w_evid, output_hidden_state_no_evid, nn.Identity()(next_state), next_checklist_with_evid, next_checklist_no_evid, reftype, reftype_err, new_item_atten_err, used_item_atten_err})
   elseif self.opt.evidence_type == 1 then -- do NOT have ref-type() and attention evidence
      reftype, next_checklist_no_evid, output_hidden_state_no_evid = attention_switch_no_evidence(self, next_h_out, prev_checklist_no_evid, agenda)

      -- Create gModule
      module = nn.gModule({prev_word, prev_word_state, agenda, goal, prev_checklist_no_evid}, {output_hidden_state_no_evid, nn.Identity()(next_state), next_checklist_no_evid, reftype})
   else
      print('Unknown evidence_type flag ' .. self.opt.evidence_type .. '.')
      os.exit()
   end

   module:getParameters():uniform(-self.opt.init_weight, self.opt.init_weight)
   return module:cuda()
end

------------------------------------------------------------------------
-- Build output vocab probabilities and compare to truth
-----------------------------------------------------------------------
function neural_checklist_model:build_output_model()
   local output_embedding = nn.Identity()()
   local true_word = nn.Identity()()

   local h2y = nn.Linear(self.opt.rnn_size, self.max_info.vocab_size)
   local dropped = nn.Dropout(self.opt.dropout)(output_embedding)
   local prediction = nn.LogSoftMax()(h2y(dropped))

   -- Evaluate predicted output vocab probabilities to true word.
   local err = nn.ClassNLLCriterion()({prediction, true_word})

   local module = nn.gModule({output_embedding, true_word}, {err})
   module:getParameters():uniform(-self.opt.init_weight, self.opt.init_weight)
   return module:cuda() 
end

-----------------------------------------------------------------------------------------
-- Create goal embedder.
-----------------------------------------------------------------------------------------
--
-- Title embedder returns two embeddings: 
--      (1) the goal embedding
--      (2) a trained projection of the goal embedding to initialize the language model
------------------------------------------------------------------------------------------
function neural_checklist_model:build_goal_embedder()
   local goal = nn.Identity()()
   local goal_lookup = LookupTableWithNulls(self.max_info.goal_vocab_size, self.opt.rnn_size)
   table.insert(self.lookups, goal_lookup)
   local sum = nn.Sum(2)(goal_lookup(goal))
   local go_rnn = nn.Linear(self.opt.rnn_size, self.opt.rnn_size)(sum)
   local module = nn.gModule({goal}, {go_rnn, sum})
   module:getParameters():uniform(-self.opt.init_weight, self.opt.init_weight)
   return module:cuda()
end

----------------------------------------------------------------------------------
-- Create the item embedder.
----------------------------------------------------------------------------------
--
-- The item embedder returns an embedding of the batched agenda.
-- Each item is a sequence of tokens. Each token's embedding is pulled from a
-- lookup table and then the embeddings are summed together.
--
-- This could be fiddled with to average the tokens, but then you lose out if the
-- item string has a lot of extra tokens. E.g., 'pulled from the vine tomatoes' would
-- have an embedding that is far less 'tomato-y' than 'tomatoes' if you average.
-----------------------------------------------------------------------------------
function neural_checklist_model:build_agenda_embedder()
   local agenda = nn.Identity()()
   local item_lookup = LookupTableWithNulls(self.max_info.item_vocab_size, self.opt.rnn_size)
   table.insert(self.lookups, item_lookup)
   local lookup2 = item_lookup(agenda)
   --emnlp doesnt run anymore local compact_items = nn.Sum(3, true)(lookup2)
   local compact_items = nn.Sum(3)(lookup2)

   local module = nn.gModule({agenda}, {compact_items})
   module:getParameters():uniform(-self.opt.init_weight, self.opt.init_weight)
   return module:cuda()
end

------------------------------------------------------------------------------
-- Reads in a set of pre-trained word embeddings for the output vocabulary.
------------------------------------------------------------------------------
-- The problem with this is that if not all the vocabulary tokens are in the
-- pre-trained set, you will have randomized embeddings for the remnants.
-----------------------------------------------------------------------------
function neural_checklist_model:read_in_embeddings(dict, embeddings, lookups)
   print('reading in')
   local embedding_file = io.open(embeddings, 'r')
   local first = true
   local counts = nil
   local seen = 0
   local used = {}
   for line in embedding_file:lines() do
      if first then
         first = false
         counts = stringx.split(line)
      else
         local split = stringx.split(line)
         local word = split[1]
         if word == '<START_RECIPE>' then
            word = '<text>'
         elseif word == '<END_RECIPE>' then
            word = '</text>'
         elseif word == '<LINE_BREAK>' then
            word = '\n'
         end
         index = dict.symbol_to_index[word]
         if index ~= nil then
            seen = seen + 1
            used[word] = true
            for i=1,counts[2] do
               for _,lookup in ipairs(lookups) do
                  lookup.weight[index+1][i] = tonumber(split[i+1])
               end
            end
         end
      end
   end
   for symbol,index in pairs(dict.symbol_to_index) do
      if used[symbol] == nil then
         print(symbol)
         for i=1,counts[2] do
            for _,lookup in ipairs(lookups) do
               lookup.weight[index+1][i] = (0.1 * math.random()) - 0.05
            end
         end
      end
   end
end

--------------------------------------------------------------------------------
-- Build all the models and initializes parameters.
-- Also generates temporary state vectors and derivative vectors.
--------------------------------------------------------------------------------
-- dict: the output vocabulary dictionary. Used if reading in word embeddings.
--------------------------------------------------------------------------------
function neural_checklist_model:build_models(dict)
   self.lookups = {} -- Set of all LookupTablesWithNulls so that the null value
                     -- can be reset at each step.
   self.model = self:build_model()
   self.embedder = self:build_agenda_embedder()
   self.outputter = self:build_output_model()
   self.goal_encoders = {}
   self.goal_paramxs = {}
   self.goal_paramdxs = {}
   for l=1,self.opt.num_layers do
      local goal_encoder = self:build_goal_embedder()
      table.insert(self.goal_encoders, goal_encoder)
      local goal_paramx, goal_paramdx = goal_encoder:getParameters()
      table.insert(self.goal_paramxs, goal_paramx)
      table.insert(self.goal_paramdxs, goal_paramdx)
   end

   self.paramx, self.paramdx = self.model:getParameters()
   self.item_paramx, self.item_paramdx = self.embedder:getParameters()
   self.out_paramx, self.out_paramdx = self.outputter:getParameters()

   self.num_steps = self.max_info.num_words
   if self.opt.embeddings ~= '' then
      print('reading in embeddings...')
      self:read_in_embeddings(dict, self.opt.embeddings, self.lookups)
      print('done')
   end


   self.final_criterion = nn.MSECriterion()
   self.final_criterion = self.final_criterion:cuda()

   -- create temp variables
   local temp_vars = {state={}, checklist_with_evid={}, checklist_no_evid={}, dev_checklist_no_evid={}, dev_checklist_with_evid={},
                  output_embedding={}, deriv={}, dev_state={},  
                      item_deriv={}, in_goal_deriv={}, goal_state={}, 
                      dev_goal_state={}, goal_deriv={}}
   temp_vars.items = torch.zeros(self.opt.batch_size, self.max_info.num_items, self.opt.rnn_size):contiguous():float():cuda()
   temp_vars.dev_items = torch.zeros(1, self.max_info.num_items, self.opt.rnn_size):contiguous():float():cuda()
   for i=0,self.num_steps do
      temp_vars.output_embedding[i] = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
      if self.opt.rnn_type == 'lstm' then
         temp_vars.state[i] = {}
         temp_vars.dev_state[i] = {}
         temp_vars.checklist_with_evid[i] = torch.zeros(self.opt.batch_size, self.max_info.num_items):contiguous():float():cuda()
         temp_vars.dev_checklist_with_evid[i] = torch.zeros(1, self.max_info.num_items):contiguous():float():cuda()
         temp_vars.checklist_no_evid[i] = torch.zeros(self.opt.batch_size, self.max_info.num_items):contiguous():float():cuda()
         temp_vars.dev_checklist_no_evid[i] = torch.zeros(1, self.max_info.num_items):contiguous():float():cuda()
         for d=1, self.opt.num_layers do
            temp_vars.state[i][(2*d) - 1] = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
            temp_vars.dev_state[i][(2*d) - 1] = torch.zeros(1, self.opt.rnn_size):contiguous():float():cuda()
            temp_vars.state[i][(2*d)] = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
            temp_vars.dev_state[i][(2*d)] = torch.zeros(1, self.opt.rnn_size):contiguous():float():cuda()
         end
      elseif self.opt.num_layers == 1 then
         temp_vars.checklist_with_evid[i] = torch.zeros(self.opt.batch_size, self.max_info.num_items):contiguous():float():cuda()
         temp_vars.dev_checklist_with_evid[i] = torch.zeros(1, self.max_info.num_items):contiguous():float():cuda()
         temp_vars.checklist_no_evid[i] = torch.zeros(self.opt.batch_size, self.max_info.num_items):contiguous():float():cuda()
         temp_vars.dev_checklist_no_evid[i] = torch.zeros(1, self.max_info.num_items):contiguous():float():cuda()
         temp_vars.state[i] = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
         temp_vars.dev_state[i] = torch.zeros(1, self.opt.rnn_size):contiguous():float():cuda()
         temp_vars.state[i] = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
         temp_vars.dev_state[i] = torch.zeros(1, self.opt.rnn_size):contiguous():float():cuda()
      else
         temp_vars.checklist_with_evid[i] = torch.zeros(self.opt.batch_size, self.max_info.num_items):contiguous():float():cuda()
         temp_vars.dev_checklist_with_evid[i] = torch.zeros(1, self.max_info.num_items):contiguous():float():cuda()
         temp_vars.checklist_no_evid[i] = torch.zeros(self.opt.batch_size, self.max_info.num_items):contiguous():float():cuda()
         temp_vars.dev_checklist_no_evid[i] = torch.zeros(1, self.max_info.num_items):contiguous():float():cuda()
         temp_vars.state[i] = {}
         temp_vars.dev_state[i] = {}
         for d=1, self.opt.num_layers do
            temp_vars.state[i][d] = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
            temp_vars.dev_state[i][d] = torch.zeros(1, self.opt.rnn_size):contiguous():float():cuda()
         end
      end
   end

   temp_vars.item_deriv = torch.zeros(self.opt.batch_size, self.max_info.num_items, self.opt.rnn_size):contiguous():float():cuda()
   temp_vars.in_goal_deriv = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
   temp_vars.output_embedding_deriv = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
   if self.opt.rnn_type == 'lstm' then
      temp_vars.deriv = {}
      temp_vars.checklist_no_evid_deriv = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
      temp_vars.checklist_with_evid_deriv = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
      for d=1, self.opt.num_layers do
         temp_vars.deriv[(2*d) - 1] = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
         temp_vars.deriv[(2*d)] = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
      end
   elseif self.opt.num_layers == 1 then
      temp_vars.checklist_no_evid_deriv = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
      temp_vars.checklist_with_evid_deriv = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
      temp_vars.deriv = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
   else
      temp_vars.deriv = {}
      temp_vars.checklist_no_evid_deriv = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
      temp_vars.checklist_with_evid_deriv = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
      for d=1, self.opt.num_layers do
         temp_vars.deriv[d] = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
      end
   end
      if self.opt.rnn_type == 'lstm' then
         temp_vars.goal_deriv = {}
         for d=1, (2*self.opt.num_layers) do
            temp_vars.goal_deriv[d] = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
         end
      elseif self.opt.num_layers == 1 then
         temp_vars.goal_deriv = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
      else
         temp_vars.goal_deriv = {}
         for d=1, self.opt.num_layers do
            temp_vars.goal_deriv[d] = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()
         end
      end

   temp_vars.dummy_reftype_deriv = torch.zeros(self.opt.batch_size, 3):cuda()
   temp_vars.ones = torch.ones(self.opt.batch_size, 1):cuda()
   temp_vars.one = torch.ones(self.opt.batch_size):cuda()
   temp_vars.dev_one = torch.ones(1):cuda()
   temp_vars.dev_ones = torch.ones(1, 1):cuda()
   temp_vars.zeros = torch.zeros(self.opt.batch_size, self.opt.rnn_size):contiguous():float():cuda()

   self.temp_vars = temp_vars

   self.avg_output = 0
   self.one = torch.CudaTensor(1):fill(1)
end

function neural_checklist_model:renorm(data, th)
    local size = data:size(1)
    for i = 2, size do
        local norm = data[i]:norm()
        if norm > th then
            data[i]:div(norm/th)
        end
    end
end

------------------------------------------------------
-- Updates the model paramters.
------------------------------------------------------
function neural_checklist_model:shrinkAndUpdateParams()
   self.norm_dw = self.paramdx:norm()
   if self.norm_dw > self.opt.max_lm_grad_norm then
      local shrink_factor = self.opt.max_lm_grad_norm / self.norm_dw
      self.paramdx:mul(shrink_factor)
   end
   self.paramx:add(self.paramdx:mul(-self.opt.learningRate))

   self.item_norm_dw = self.item_paramdx:norm()
   if self.item_norm_dw > self.opt.max_item_grad_norm then
      local shrink_factor = self.opt.max_item_grad_norm / self.item_norm_dw
      self.item_paramdx:mul(shrink_factor)
   end
   self.item_paramx:add(self.item_paramdx:mul(-self.opt.learningRate))
   self.out_norm_dw = self.out_paramdx:norm()
   if self.out_norm_dw > self.opt.max_lm_grad_norm then
      local shrink_factor = self.opt.max_lm_grad_norm / self.out_norm_dw
      self.out_paramdx:mul(shrink_factor)
   end
   self.out_paramx:add(self.out_paramdx:mul(-self.opt.learningRate))
   if self.opt.encode_goal then
      for l=1,self.opt.num_layers do
         self.goal_norm_dw = self.goal_paramdxs[l]:norm()
         if self.goal_norm_dw > self.opt.max_goal_grad_norm then
            local shrink_factor = self.opt.max_goal_grad_norm / self.goal_norm_dw
            self.goal_paramdxs[l]:mul(shrink_factor)
         end
         self.goal_paramxs[l]:add(self.goal_paramdxs[l]:mul(-self.opt.learningRate))
      end
   end
end


-----------------------------------------------
-- Reset temporary state vectors.
-----------------------------------------------
function neural_checklist_model:reset_state()
   self.temp_vars.items:zero()
   self.temp_vars.dev_items:zero()
   if self.opt.rnn_type == 'lstm' then
      for i=0,self.num_steps do
         for d=1, (2*self.opt.num_layers) do
            self.temp_vars.state[i][d]:zero()
            self.temp_vars.dev_state[i][d]:zero()
         end
      end
   elseif self.opt.num_layers == 1 then
      for i=0,self.num_steps do
         self.temp_vars.state[i]:zero()
         self.temp_vars.dev_state[i]:zero()
      end
   else
      for i=0,self.num_steps do
         for d=1, self.opt.num_layers do
            self.temp_vars.state[i][d]:zero()
            self.temp_vars.dev_state[i][d]:zero()
         end
      end
   end
end

-----------------------------------------------
-- Reset temporary gradient vectors.
-----------------------------------------------
function neural_checklist_model:reset_deriv()
   if self.opt.rnn_type == 'lstm' then
      for d=1, 2*self.opt.num_layers do
         self.temp_vars.deriv[d]:zero()
         self.temp_vars.goal_deriv[d]:zero()
      end
   elseif self.opt.num_layers == 1 then
      self.temp_vars.deriv:zero()
      self.temp_vars.goal_deriv:zero()
   else
      for d=1, self.opt.num_layers do
         self.temp_vars.deriv[d]:zero()
         self.temp_vars.goal_deriv[d]:zero()
      end
   end
end

--------------------------------------------------------------------------
-- Performs forward pass assuming minibatch size of 1.
-- Uses temporary variables that have 'dev' in them that are
-- sized appropriately.
--
-- Part of the neural checklist model code requires specific computations
-- that assume minibatched instances.
--------------------------------------------------------------------------
function neural_checklist_model:dev_forward_pass(state)
   -- Reset state variables
   self:reset_state()

   -- Re-zero out the embedding for the "null" token.
   -- Null tokens occur when, for example, agendas are batched that have
   -- different numbers of tokens for agenda items.
   for _,lookup in ipairs(self.lookups) do
      lookup:resetNullWeight()
   end

   -- Remove dropouts for development evaluation
   for _,drop in ipairs(self.dropouts) do
      drop:setp(0.0)
   end

   self.avg_output = 0

   -- Embed agenda
   local item_tmp = self.embedder:forward(state.agenda)
   self.temp_vars.dev_items:narrow(2, 1, state.batch_len[3]):copy(item_tmp)

   -- Embed goal
   local rnn_start = {}
   local embedded_goal = {}
      for l=1,self.opt.num_layers do
         local rnn_part, goal_part = unpack(self.goal_encoders[l]:forward(state.goal))
         table.insert(embedded_goal, goal_part)
         table.insert(rnn_start, rnn_part)

         if self.opt.rnn_type == 'lstm' then
            self.temp_vars.dev_state[1][2*l]:copy(rnn_start[l])
         elseif self.opt.num_layers == 1 then
            self.temp_vars.dev_state[1]:copy(rnn_start[l])
         else
            self.temp_vars.dev_state[1][l]:copy(rnn_start[l])
         end
      end

   -- Zero out checklist temporary variables.
   for s=1,state.batch_len[1] do
      self.temp_vars.dev_checklist_with_evid[s]:resize(1, state.batch_len[3]):zero()
      self.temp_vars.dev_checklist_no_evid[s]:resize(1, state.batch_len[3]):zero()
   end

   -- Perform forward pass through text token-by-token
   local other_err = 0
   for s=2,state.batch_len[1] do
      local word_err = nil
      local next_state = nil
      local reftype = nil
      local reftype_err = nil
      local new_item_atten_err = nil
      local used_item_atten_err = nil
      local next_checklist_with_evid = nil
      local next_checklist_no_evid = nil
      local output_hidden_state = nil
      local output_hidden_state_w_evid = nil
      local output_hidden_state_no_evid = nil

      if self.opt.evidence_type == 0 then -- have ref-type() and attention evidence
         output_hidden_state_w_evid, output_hidden_state_no_evid, next_state, next_checklist_with_evid, next_checklist_no_evid, reftype, reftype_err, new_item_atten_err, used_item_atten_err = unpack(self.model:forward({state.text[s-1], self.temp_vars.dev_state[s-1], self.temp_vars.dev_items:narrow(2, 1, state.batch_len[3]), embedded_goal[1], self.temp_vars.dev_checklist_with_evid[s-1], self.temp_vars.dev_checklist_no_evid[s-1], state.ref_types[s], state.true_new_item_atten[s], state.true_used_item_atten[s]}))
         output_hidden_state = output_hidden_state_w_evid

         -- Copy checklist information for next step
         self.temp_vars.dev_checklist_with_evid[s]:copy(next_checklist_with_evid)
         self.temp_vars.dev_checklist_no_evid[s]:copy(next_checklist_no_evid)

         -- Accumulate errors for printout
         other_err = other_err + reftype_err[1]
         other_err = other_err + new_item_atten_err[1]
         other_err = other_err + used_item_atten_err[1]
      elseif self.opt.evidence_type == 1 then -- do NOT have ref-type() and attention evidence
         output_hidden_state_no_evid, next_state, next_checklist_no_evid, reftype = unpack(self.model:forward({state.text[s-1], self.temp_vars.dev_state[s-1], self.temp_vars.dev_items:narrow(2, 1, state.batch_len[3]), embedded_goal[1], self.temp_vars.dev_checklist_no_evid[s-1]}))
         output_hidden_state = output_hidden_state_no_evid

         -- Copy checklist information for next step
         self.temp_vars.dev_checklist_no_evid[s]:copy(next_checklist_no_evid)
      else
         print('Unknown evidence type ' .. self.opt.evidence_type .. '.')
         os.exit()
      end

      -- Get error on output vocabulary probabilities.
      -- (This also implicitly generates the output vocabulary probabilities.)
      word_err = self.outputter:forward({output_hidden_state, state.text[s]})

      -- Copy next state information
      if self.opt.rnn_type == 'lstm' then
         model_utils.copy_table(self.temp_vars.dev_state[s], next_state)
      elseif self.opt.num_layers == 1 then
         self.temp_vars.dev_state[s]:copy(next_state)
      else
         model_utils.copy_table(self.temp_vars.dev_state[s], next_state)
      end
      
      -- Accumulate errors for printout
      self.avg_output = self.avg_output + word_err[1]
   end

   -- If no evidence we need a loss on agenda items that have not been used.
   -- This is not necessary (and would be strange) for the case when there is
   -- evidence on when agenda items are used.
   if self.opt.evidence_type == 1 then
      self.temp_vars.dev_ones:resize(1, state.batch_len[3]):fill(self.opt.end_loss_mul)
      self.temp_vars.dev_checklist_no_evid[state.batch_len[1] ]:mul(self.opt.end_loss_mul)
      local end_err = self.final_criterion:forward(self.temp_vars.dev_checklist_no_evid[state.batch_len[1] ], self.temp_vars.dev_ones)
      other_err = other_err + end_err
   end

   cutorch.synchronize()
   return self.avg_output, other_err
end

---------------------------------------------------------
-- Performs a forward pass over a minibatch (state).
---------------------------------------------------------
function neural_checklist_model:forward_pass(state)
   -- Reset state variables
   self:reset_state()

   -- Re-zero out the embedding for the "null" token.
   -- Null tokens occur when, for example, agendas are batched that have
   -- different numbers of tokens for agenda items.
   for _,lookup in ipairs(self.lookups) do
      lookup:resetNullWeight()
   end

   -- Add dropout weights, if provided.
   for _,drop in ipairs(self.dropouts) do
      drop:setp(self.opt.dropout)
   end

   -- Embed agenda 
   local item_tmp = self.embedder:forward(state.agenda)
   self.temp_vars.items:narrow(2, 1, state.batch_len[3]):copy(item_tmp)

   -- Embed goal
   self.rnn_start = {}
   self.embedded_goal = {}
   for l=1,self.opt.num_layers do
      local rnn_part, goal_part = unpack(self.goal_encoders[l]:forward(state.goal))
      table.insert(self.embedded_goal, goal_part)
      table.insert(self.rnn_start, rnn_part)

      if self.opt.rnn_type == 'lstm' then
         self.temp_vars.state[1][2*l]:copy(self.rnn_start[l])
      elseif self.opt.num_layers == 1 then
         self.temp_vars.state[1]:copy(self.rnn_start[l])
      else
         self.temp_vars.state[1][l]:copy(self.rnn_start[l])
      end
   end

   -- Zero out checklist temporary variables.
   for s=1,state.batch_len[1] do
      self.temp_vars.checklist_with_evid[s]:resize(self.opt.batch_size, state.batch_len[3]):zero()
      self.temp_vars.checklist_no_evid[s]:resize(self.opt.batch_size, state.batch_len[3]):zero()
   end

   -- Perform forward pass through text token-by-token
   local other_err = 0
   self.avg_output = 0
   for s=2,state.batch_len[1] do
      local word_err = nil
      local next_state = nil
      local reftype = nil
      local reftype_err = nil
      local new_item_atten_err = nil
      local used_item_atten_err = nil
      local next_checklist_with_evid = nil
      local next_checklist_no_evid = nil
      local output_hidden_state = nil
      local output_hidden_state_w_evid = nil
      local output_hidden_state_no_evid = nil

      if self.opt.evidence_type == 0 then -- have ref-type() and attention evidence
         output_hidden_state_w_evid, output_hidden_state_no_evid, next_state, next_checklist_with_evid, next_checklist_no_evid, reftype, reftype_err, new_item_atten_err, used_item_atten_err = unpack(self.model:forward({state.text[s-1], self.temp_vars.state[s-1], self.temp_vars.items:narrow(2, 1, state.batch_len[3]), self.embedded_goal[1], self.temp_vars.checklist_with_evid[s-1], self.temp_vars.checklist_no_evid[s-1], state.ref_types[s], state.true_new_item_atten[s], state.true_used_item_atten[s]}))
         output_hidden_state = output_hidden_state_w_evid

         -- Copy checklist information for next step
         self.temp_vars.checklist_with_evid[s]:copy(next_checklist_with_evid)
         self.temp_vars.checklist_no_evid[s]:copy(next_checklist_no_evid)

         -- Accumulate errors for printout
         other_err = other_err + reftype_err[1]
         other_err = other_err + new_item_atten_err[1]
         other_err = other_err + used_item_atten_err[1]
      elseif self.opt.evidence_type == 1 then -- do NOT have ref-type() and attention evidence
         output_hidden_state_no_evid, next_state, next_checklist_no_evid, reftype = unpack(self.model:forward({state.text[s-1], self.temp_vars.state[s-1], self.temp_vars.items:narrow(2, 1, state.batch_len[3]), self.embedded_goal[1], self.temp_vars.checklist_no_evid[s-1]}))
         output_hidden_state = output_hidden_state_no_evid

         -- Copy checklist information for next step
         self.temp_vars.checklist_no_evid[s]:copy(next_checklist_no_evid)
      else
         print('Unknown evidence type ' .. self.opt.evidence_type .. '.')
         os.exit()
      end

      -- Get error on output vocabulary probabilities.
      -- (This also implicitly generates the output vocabulary probabilities.)
      word_err = self.outputter:forward({output_hidden_state, state.text[s]})
      self.temp_vars.output_embedding[s]:copy(output_hidden_state)

      -- Copy next state information
      local next_h = nil
      if self.opt.rnn_type == 'lstm' then
         model_utils.copy_table(self.temp_vars.state[s], next_state)
         next_h = next_state[2*self.opt.num_layers]
      elseif self.opt.num_layers == 1 then
         self.temp_vars.state[s]:copy(next_state)
         next_h = next_state
      else
         model_utils.copy_table(self.temp_vars.state[s], next_state)
         next_h = next_state[self.opt.num_layers]
      end
      if word_err[1] ~= word_err[1] then
         print('NaN error. :(')
         os.exit(1)
      end

      -- Accumulate errors for printout
      self.avg_output = self.avg_output + word_err[1]
   end

   -- If no evidence we need a loss on agenda items that have not been used.
   -- This is not necessary (and would be strange) for the case when there is
   -- evidence on when agenda items are used.
   if self.opt.evidence_type == 1 then
      self.temp_vars.ones:resize(self.opt.batch_size, state.batch_len[3]):fill(self.opt.end_loss_mul)
      self.temp_vars.checklist_no_evid[state.batch_len[1] ]:mul(self.opt.end_loss_mul)
      local end_err = self.final_criterion:forward(self.temp_vars.checklist_no_evid[state.batch_len[1] ], self.temp_vars.ones)
      other_err = other_err + end_err
   end

   cutorch.synchronize()
   return self.avg_output
end

-----------------------------------------------------------
-- Backward pass for a minibatch (state)
-----------------------------------------------------------
function neural_checklist_model:backward_pass(state)
   -- Zero out gradients.
   self.paramdx:zero()
   self.item_paramdx:zero()
   self.out_paramdx:zero()
   if self.opt.encode_goal then
      for _,paramdx in pairs(self.goal_paramdxs) do
         paramdx:zero()
      end
   end

   -- Resize checklist gradient vectors.
   self.temp_vars.checklist_no_evid_deriv:resize(self.opt.batch_size, state.batch_len[3], 1):zero()
   self.temp_vars.checklist_with_evid_deriv:resize(self.opt.batch_size, state.batch_len[3], 1):zero()

   -- Resize goal and agenda gradient vectors.
   self.temp_vars.item_deriv:resize(self.opt.batch_size, state.batch_len[3], self.opt.rnn_size):zero():contiguous():float():cuda()
   self.temp_vars.in_goal_deriv:zero()
   self.temp_vars.zeros:zero()
   self.temp_vars.output_embedding_deriv:zero()

   -- Reset other gradient vectors.
   self:reset_deriv()

   if self.opt.evidence_type == 1 then
      local start_state_deriv = self.final_criterion:backward(self.temp_vars.checklist_no_evid[state.batch_len[1] ], self.temp_vars.ones)
      self.temp_vars.checklist_no_evid_deriv:copy(start_state_deriv)
   end

   -- Backward pass token-by-token
   for s = state.batch_len[1], 2, -1 do
      self.outputter:forward({self.temp_vars.output_embedding[s], state.text[s]})
      local output_tmp = self.outputter:backward({self.temp_vars.output_embedding[s], state.text[s]}, self.one)
      self.temp_vars.output_embedding_deriv:copy(output_tmp[1])
      local tmp = nil
      if self.opt.evidence_type == 0 then -- have ref-type() and attention evidence
         self.model:forward({state.text[s-1], self.temp_vars.state[s-1], self.temp_vars.items:narrow(2, 1, state.batch_len[3]), self.embedded_goal[1], self.temp_vars.checklist_with_evid[s-1], self.temp_vars.checklist_no_evid[s-1], state.ref_types[s], state.true_new_item_atten[s], state.true_used_item_atten[s]})
         tmp = self.model:backward({state.text[s-1], self.temp_vars.state[s-1], self.temp_vars.items:narrow(2, 1, state.batch_len[3]), self.embedded_goal[1], self.temp_vars.checklist_with_evid[s-1], self.temp_vars.checklist_no_evid[s-1], state.ref_types[s], state.true_new_item_atten[s], state.true_used_item_atten[s]}, {self.temp_vars.output_embedding_deriv, self.temp_vars.zeros, self.temp_vars.deriv, self.temp_vars.checklist_with_evid_deriv, self.temp_vars.checklist_no_evid_deriv, self.temp_vars.dummy_reftype_deriv, self.one, self.one, self.one})

         self.temp_vars.checklist_with_evid_deriv:copy(tmp[5])
         -- We don't update the checklist without evidence derivative since that output is not used in training.
         -- It is only for generation when we want to avoid any evidence.

      elseif self.opt.evidence_type == 1 then -- do NOT have ref-type() and attention evidence
         self.model:forward({state.text[s-1], self.temp_vars.state[s-1], self.temp_vars.items:narrow(2, 1, state.batch_len[3]), self.embedded_goal[1], self.temp_vars.checklist_no_evid[s-1]})
         tmp = self.model:backward({state.text[s-1], self.temp_vars.state[s-1], self.temp_vars.items:narrow(2, 1, state.batch_len[3]), self.embedded_goal[1], self.temp_vars.checklist_no_evid[s-1]}, {self.temp_vars.output_embedding_deriv, self.temp_vars.deriv, self.temp_vars.checklist_no_evid_deriv, self.temp_vars.dummy_reftype_deriv})

         self.temp_vars.checklist_no_evid_deriv:copy(tmp[5])
      else
         print('Unknown evidence type ' .. self.opt.evidence_type .. '.')
         os.exit()
      end
      if self.opt.rnn_type ~= 'lstm' and self.opt.num_layers == 1 then
         self.temp_vars.deriv:copy(tmp[2])
      else
         model_utils.copy_table(self.temp_vars.deriv, tmp[2])
      end
      self.temp_vars.item_deriv:add(tmp[3])
      self.temp_vars.in_goal_deriv:add(tmp[4])
   end

   -- The gradient for the hidden state at the end of the backward pass is passed to the goal encoder.
   if self.opt.rnn_type == 'lstm' then
      for l=1,self.opt.num_layers do
         self.temp_vars.goal_deriv[l]:copy(self.temp_vars.deriv[2*l])
      end
   elseif self.opt.num_layers == 1 then
      self.temp_vars.goal_deriv:copy(self.temp_vars.deriv)
   else
      for l=1,self.opt.num_layers do
         self.temp_vars.goal_deriv[l]:copy(self.temp_vars.deriv[l])
      end
   end

   -- Backward pass through the goal encoder
   if self.opt.rnn_type == 'lstm' then
      self.goal_encoders[1]:backward(state.goal, self.temp_vars.goal_deriv[1])
   elseif self.opt.num_layers == 1 then
      self.goal_encoders[1]:backward(state.goal, {self.temp_vars.goal_deriv, self.temp_vars.in_goal_deriv})
   else
      for l=1,self.opt.num_layers do
         self.goal_encoders[l]:backward(state.goal, self.temp_vars.goal_deriv[l])
      end
   end

   -- Backward pass through the agenda embedder
   self.embedder:backward(state.agenda, self.temp_vars.item_deriv)
   cutorch.synchronize()

   -- Update parameters.
   self:shrinkAndUpdateParams()
end

------------------------------------------------------
-- Save the models.
------------------------------------------------------
function neural_checklist_model:save_model(epoch, info)
   -- Reset any parameter weights for the null token.
   for _,lookup in ipairs(self.lookups) do
      lookup:resetNullWeight()
   end

   -- Remove dropout so the model can be used for generation.
   for _,drop in ipairs(self.dropouts) do
      drop:setp(0.0)
   end

   -- Create file prefix with opt.info version info text.
   local outfile_name = self.opt.checkpoint_dir .. 'neural_checklist_model.' .. info

   torch.save(outfile_name .. '.model.ep' .. epoch, self.model)
   torch.save(outfile_name .. '.embed.ep' .. epoch, self.embedder)
   torch.save(outfile_name .. '.outputter.ep' .. epoch, self.outputter)
   for l=1,self.opt.num_layers do
      torch.save(outfile_name .. '.goal' .. l .. '.ep' .. epoch, self.goal_encoders[l])
   end
   torch.save(outfile_name .. '.opt', self.opt)
   torch.save(outfile_name .. '.max_info', self.max_info)
end

return neural_checklist_model

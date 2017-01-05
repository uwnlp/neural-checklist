require 'cutorch'

local RecipeDataMinibatchLoader = require 'RecipeDataMinibatchLoader'
local neural_checklist_model = require 'neural_checklist_model'
local stringx = require('pl.stringx')

local cmd = torch.CmdLine()
cmd:option('-gpuidx', 1, 'Index of GPU on which job should be executed.')
cmd:option('-train_data_dir', '', 'Directory with the training data')
cmd:option('-data_dir', '', 'Directory with the data to generate recipes from')
cmd:option('-info', '', 'Model file name information')
cmd:option('-checkpoint_dir', '', 'output directory where checkpoints are written')
cmd:option('-epoch', 1, '')
cmd:option('-max_beams', 100, '')
cmd:option('-force_different_first_tokens', false, 'Force the first token of each sentence to be different')
cmd:option('-use_checklist_plus', false, 'Use checklist+ model')
cmd:option('-data_file_info', 'v1.', 'Data file name information')
cmd:option('-max_length', 200, '')
cmd:option('-use_sampling', false, 'Use sampling during beam search')
cmd:option('-temperature', 1, '')
cmd:option('-verbose', false, '')
cmd:option('-beam_size',  2, 'beam size')
cmd:option('-outFile', '', 'Output file')
cmd:option('-cpu', false, 'Use CPU (doesnt really work)')

cmd:text()
opt = cmd:parse(arg)
torch.manualSeed(123)
cutorch.setDevice(opt.gpuidx)

function get_predictions(model, data_loader, max_info)
   local out = nil
   local out_true = nil
   local out_pred = nil
   if opt.outFile ~= '' then
      out = io.open(opt.outFile, 'w')
      out_true = io.open(opt.outFile .. '.truth', 'w')
      out_pred = io.open(opt.outFile .. '.pred', 'w')
   end

   local dict = data_loader.dict
   local item_dict = data_loader.item_dict
   local goal_dict = data_loader.goal_dict
   local end_text = dict.symbol_to_index['</text>']
   local start_text = dict.symbol_to_index['<text>']

   local ps = nil
   if model.opt.rnn_type == 'lstm' then
      ps = {}
      for d = 1, model.opt.num_layers do
         if opt.cpu then
            ps[(2*d) - 1] = torch.zeros(1, model.opt.rnn_size) -- for prediction
            ps[(2*d)] = torch.zeros(1, model.opt.rnn_size) -- for prediction
         else
            ps[(2*d) - 1] = torch.zeros(1, model.opt.rnn_size):cuda() -- for prediction
            ps[(2*d)] = torch.zeros(1, model.opt.rnn_size):cuda() -- for prediction
         end
      end
   elseif model.opt.num_layers == 1 then
      if opt.cpu then
         ps = torch.zeros(1, model.opt.rnn_size) -- for prediction
      else
         ps = torch.zeros(1, model.opt.rnn_size):cuda() -- for prediction
      end
   else
      if opt.cpu then
         ps = {}
         for d = 1, model.opt.num_layers do
            ps[d] = torch.zeros(1, model.opt.rnn_size) -- for prediction
         end
      else
         ps = {}
         for d = 1, model.opt.num_layers do
            ps[d] = torch.zeros(1, model.opt.rnn_size):cuda() -- for prediction
         end
      end
   end

   for d=1,data_loader.nvalid do
      print('example: ' .. d)
      local text, goal, agenda, batch_len, ref_types, true_new_item_atten, true_used_item_atten = data_loader:next_batch(2)
      local state = {text = text,
                     goal = goal,
                     agenda = agenda,
                     batch_len = batch_len}
--   if step_idx == 1 then
      local goal = ''
      for v=1,state.goal:size(2) do
         local token = state.goal[1][v]
         if token ~= 0 then
            goal = goal .. goal_dict.index_to_symbol[token ] .. ' '
         end
      end
      if out ~= nil then
         out:write(stringx.strip(goal) .. '\n')
      end
      print('goal: ' .. goal .. '\n')

      local agenda = ''
      local item_table = {}
      local num_agenda = 1
      for i=num_agenda,state.agenda:size(2) do
         local item = ''
         for j=1,state.agenda:size(3) do
            local word = state.agenda[1][i][j]
            if word ~= max_info.pad then
               agenda = agenda .. item_dict.index_to_symbol[word] .. ' '
               item = item .. item_dict.index_to_symbol[word] .. ' '
            else
               break
            end
         end
         table.insert(item_table, item)
         agenda = agenda .. '\n'
      end
      if out ~= nil then
         out:write(stringx.strip(stringx.replace(agenda, '\n', '\t')) .. '\n')
      end
      print(agenda .. '\n')
      local prediction, beam = model:get_prediction(ps, state, dict, item_dict, max_info.vocab_size, nil, false)
      io.write('Prediction:\n' .. prediction .. '\n')
      local item_weights = torch.ones(state.batch_len[3], 1):contiguous():float():cuda()
      local need_redo = false
      local all_missed = true
      local num_used = 0
      if opt.use_checklist_plus then
         for ing=1,state.batch_len[3] do
            if beam.used_item_idxs[ing] == nil then
               if item_table[ing]:find('<unk>') == nil then
                  need_redo = true
                  item_weights[ing][1] = 2
               end
            else
               num_used = num_used + 1
               all_missed = false
            end
         end
      end
      local original_prediction = stringx.replace(prediction, '  ', ' ')
      if string.len(original_prediction) > 8 and string.sub(original_prediction, -7) == '</text>' then
         original_prediction = string.sub(original_prediction, 1, string.len(original_prediction) - 8)
      end
         local prev_prediction = prediction
         local prev_beam = beam
      while need_redo do
      --while need_redo and not all_missed do EMNLP
         print(item_weights:t())

         prediction, beam = model:get_prediction(ps, state, dict, item_dict, max_info.vocab_size, item_weights, false)

         io.write('new prediction:\n' .. prediction .. '\n')
         need_redo = false
         all_missed = true
         local new_num_used = 0
         for ing=1,state.batch_len[3] do
            if beam.used_item_idxs[ing] == nil then
               if item_table[ing]:find('<unk>') == nil then
                  need_redo = true
                  item_weights[ing][1] = item_weights[ing][1] + 1
               end
            else
               new_num_used = new_num_used + 1
               all_missed = false
            end
         end
         if new_num_used <= num_used then
            prediction = prev_prediction
            beam = prev_beam
            print("Revert!  " .. num_used .. ' ' .. new_num_used)
            break
         else
            num_used = new_num_used
            prev_prediction = prediction
            prev_beam = beam
         end
      end
         if out ~= nil then
            out:write(original_prediction .. '\nEND RECIPE\n')
         end
         original_prediction = stringx.replace(original_prediction, '_', ' ')
         original_prediction = stringx.replace(original_prediction, '  ', ' ')
         if out ~= nil then
            out_pred:write(stringx.strip(stringx.replace(original_prediction, '\n', ' ')) .. '\n')
         end
      if opt.use_checklist_plus then
         prediction = stringx.replace(prediction, '  ', ' ')
         if string.len(prediction) > 8 and string.sub(prediction, -7) == '</text>' then
            prediction = string.sub(prediction, 1, string.len(prediction) - 8)
         end
         if out ~= nil then
            out:write(prediction .. '\nEND RECIPE\n')
            prediction = stringx.replace(prediction, '_', ' ')
            prediction = stringx.replace(prediction, '  ', ' ')
            out_pred:write(stringx.strip(stringx.replace(prediction, '\n', ' ')) .. '\n')
         end
         print((1.0 / beam.len) * beam.prob)
      end
      io.write('Truth\n')
      local truth = ''
         for i=2,batch_len[1]-1 do
            local symbol = dict.index_to_symbol[state.text[i][1] ]
            if symbol ~= nil then
               truth = truth .. symbol .. ' '
            end
      end
      truth = stringx.replace(truth, '  ', ' ')
      truth = stringx.replace(truth, '  ', ' ')
      print(truth)
      if out ~= nil then
         out:write(stringx.strip(truth) .. '\nEND RECIPE\n')
         truth = stringx.replace(truth, '_', ' ')
         out_true:write(stringx.strip(stringx.replace(truth, '\n', ' ')) .. '\n')
      end
   end
   if out ~= nil then
      out:close()
      out_pred:close()
      out_true:close()
   end
end


function load_model(max_info)
   return neural_checklist_model:load_generation_model(opt, opt.info, opt.epoch)
end

function main()

   local data_loader = RecipeDataMinibatchLoader.create(opt.train_data_dir, opt.data_dir, 1, true, opt.data_file_info, opt, false)
   local vocab_size = data_loader.vocab_size

   local max_info = {}
   max_info.num_words = data_loader.max_num_words
   max_info.vocab_size = vocab_size
   max_info.item_vocab_size = data_loader.item_vocab_size
   max_info.goal_vocab_size = data_loader.goal_vocab_size
   max_info.pad = data_loader.pad
   max_info.num_items = data_loader.max_num_items
   max_info.item_length = data_loader.max_item_length
   max_info.goal_length = data_loader.max_goal_length
   max_info.text_length = data_loader.max_text_length

   model = load_model(max_info)

   get_predictions(model, data_loader, max_info)
end

main()

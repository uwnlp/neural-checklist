require 'torch'
require 'cutorch'
local stringx = require('pl.stringx')
local utils = require('utils.utils')

local RecipeDataMinibatchLoader = {}
RecipeDataMinibatchLoader.__index = RecipeDataMinibatchLoader

-----------------------------------------------------------------------------------------
-- The RecipeDataMinibatchLoader loads recipe data from torch files
-- built from build_recipe_mats.lua into minibatches for training/testing.

-- Each minibatch i of size b is represented by a set of matrices:
--       self.text_batches[i]:      A (max_text_length x b) matrix representing the output 
--                                  recipe text of the minibatch instances
--       self.goal_batches[i]:      A (b x max_goal_length_in_tokens) matrix representing the
--                                  goals (titles) of the minibatch instances
--       self.items_batches[i]:     A (b x agenda_length x max_item_length_in_tokens) matrix
--                                  representing the agendas (ingredient lists) of the 
--                                  minibatch instances
--       self.batch_len[i]:         A length 4 tensor representing information about the batch:
--                                     self.batch_len[i][1] = text_len
--                                     self.batch_len[i][2] = max_goal_length
--                                     self.batch_len[i][3] = curr_num_items
--                                     self.batch_len[i][4] = max_item_length
--       self.ref_type_batches[i]:  A (max_text_length x b x 3) matrix representing the true
--                                  values of the ref-type() classifier.
--       self.true_new_item_atten_batches[i]: A (max_text_length x b x agenda_length) matrix
--                                  representing the true new item attentions for each step
--       self.true_used_item_atten_batches[i]: A (max_text_length x b x agenda_length) matrix
--                                  representing the true used item attentions for each step
--
--
-- The development/test set are loaded into minibatches of size 1 with "dev_" preceeding
-- the matrices (e.g., self.dev_text_batches).
----------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------
-- Load data and create minibatches.
-----------------------------------------------------------------------------------------
--
-- data_type: 'train' or 'dev' If dev, the minibatches have size 1 and aren't randomized
-- data_dir: directory of the data to load
-- dict, item_dict, goal_dict: three dictionaries
-- rand (boolean): randomize minibatches
-- info: data file version info
-- opt: trainitem option flags
--------------------------------------------------------------------------------------------
function RecipeDataMinibatchLoader:loadData(data_type, data_dir, dict, item_dict, goal_dict, rand, info, opt)
   local offset_file = path.join(data_dir, 'offset.' .. info .. 'mat.torch')
   local full_text_file = path.join(data_dir, 'text.' .. info .. 'mat.torch')
   local full_text_to_pos_file = path.join(data_dir, 'text_to_pos.' .. info .. 'mat.torch')
   local items_file = path.join(data_dir, 'items.' .. info .. 'mat.torch')
   local item_file = path.join(data_dir, 'item.' .. info .. 'mat.torch')
   local goal_file = path.join(data_dir, 'goal.' .. info .. 'mat.torch')

   local texts = torch.load(full_text_file)
   local text_to_pos = torch.load(full_text_to_pos_file)
   local offsets = torch.load(offset_file)
   local goals = torch.load(goal_file)
   local items = torch.load(items_file)
   local item = torch.load(item_file)

   local num_texts = offsets:size(1)
   local batch_size = self.batch_size
   local num_batches_to_sort_at_once = self.num_batches_to_sort_at_once
   if data_type == 'dev' then
      batch_size = 1
   end
   local num_batches = 0
   local leftover_texts = 0

   local texts_by_num_items = {}
   local texts_by_num_items_idx = {}

   -- Some token indices we will use.
   local start_text = dict.symbol_to_index['<text>']
   local line_break = dict.symbol_to_index['\n']
   local unk = dict.symbol_to_index['<unk>']
   local goal_unk = goal_dict.symbol_to_index['<unk>']
   local item_unk = item_dict.symbol_to_index['<unk>']
   local all = dict.symbol_to_index['<ALL>']


   -- This code batches examples that have the same length and the same agenda size.
   -- It is possible that the same agenda size can be avoided, but that is not
   -- implemented here.
   -- This part of the code counts the number of batches and removes examples that will
   -- not be in batches based on their sizes.
   local sanity = 0
   if data_type == 'train' then
      for length,mat in pairs(texts) do -- for each set of texts of a particular length,
         local num_texts_of_length = mat:size(1)
         sanity = sanity + num_texts_of_length
         texts_by_num_items[length] = {}
         texts_by_num_items_idx[length] = {}
         for i=1,num_texts_of_length do -- for each text of that length,
            local offsets_idx = text_to_pos[length][i] -- get text index
            local offset_info = offsets[offsets_idx] -- and get text information
            local num_items = offset_info[5]
            -- add text into a table based on its length and agenda size (e.g., num ingredients)
            if texts_by_num_items_idx[length][num_items]  == nil then
               texts_by_num_items_idx[length][num_items] = {}
            end
            table.insert(texts_by_num_items_idx[length][num_items], i)
            texts_by_num_items[length][num_items] = (texts_by_num_items[length][num_items] or 0) + 1
         end
         for num_items, count in pairs(texts_by_num_items[length]) do -- for each set of texts of a size
                                                                      -- and num ingredients,
            -- If the number of texts of those sizes is smaller than the batch_size, the texts
            -- won't be in batches.
            if count < batch_size then
               texts_by_num_items_idx[length][num_items] = {}
               texts_by_num_items[length][num_items] = 0
               leftover_texts = leftover_texts + count
            else
               -- Otherwise, remove a random subset so that we have the correct number for equally-sized batches
               local leftover_of_length = count % batch_size
               leftover_texts = leftover_texts + leftover_of_length
               num_batches = num_batches + math.floor(count / batch_size)
               local random_ordering = torch.randperm(count)
               for i=1,leftover_of_length do
                  texts_by_num_items_idx[length][num_items][random_ordering[i]] = nil
               end
               texts_by_num_items[length][num_items] = texts_by_num_items[length][num_items] - leftover_of_length
               local new = {}
               for _,idx in pairs(texts_by_num_items_idx[length][num_items]) do
                  if idx ~= nil then
                     table.insert(new, idx)
                  end
               end
               texts_by_num_items_idx[length][num_items] = new
               if #(texts_by_num_items_idx[length][num_items]) == 0 then
                  print(count)
                  print(batch_size)
                  print(leftover_of_length)
                  os.exit()
               end
            end
         end
      end
   else -- for dev: same as above for train, but we don't ignore any texts size batch size is 1
      for length,mat in pairs(texts) do
         local num_texts_of_length = mat:size(1)
         texts_by_num_items[length] = {}
         texts_by_num_items_idx[length] = {}
         for i=1,num_texts_of_length do
            local offsets_idx = text_to_pos[length][i]
            local offset_info = offsets[offsets_idx]
            local num_items = offset_info[5]
            if texts_by_num_items_idx[length][num_items]  == nil then
               texts_by_num_items_idx[length][num_items] = {}
            end
            table.insert(texts_by_num_items_idx[length][num_items], i)
            texts_by_num_items[length][num_items] = (texts_by_num_items[length][num_items] or 0) + 1
         end
         for num_items, count in pairs(texts_by_num_items[length]) do
            num_batches = num_batches + count
         end
      end
   end
   print('num batches = ' .. num_batches)
   print('excludes ' .. leftover_texts .. ' texts')
   print('uses ' .. (num_batches * batch_size) .. ' texts')
   if data_type == 'train' then
      self.ntrain = num_batches
      self.split_sizes[1] = num_batches
   elseif data_type == 'dev' then
      self.nvalid = num_batches
      self.split_sizes[2] = num_batches
   end

   -- Generate random batch ordering.
   local random_text_ordering = nil
   if rand then
      random_batch_ordering = torch.randperm(num_batches)
      self.random_batch_ordering = random_batch_ordering
   elseif data_type == 'train' then
      random_batch_ordering = torch.range(1,num_batches)
      self.random_batch_ordering = random_batch_ordering
   else
      random_batch_ordering = torch.range(1,num_batches)
      self.dev_random_batch_ordering = random_batch_ordering
   end

   -- creatitem batch info
   local text_batches = {} -- holds batches of text text
   local goal_batches = {} -- holds batches of goals
   local items_batches = {} -- holds batches of ingredients
   local batch_len = {} -- holds information for each batch
   local ref_type_batches = {} -- holds true values for ref-type()
   local true_new_item_atten_batches = {} -- holds batches of true values for new ingredient attentions
   local true_used_item_atten_batches = {} -- holds batches of true values for used ingredient attentions

   local batch_counter = 1

   -- Create batches.
   -- This code also identifies the maximum lengths, num ingredients, etc. which is used to
   -- initialize temporary structures for model trainitem.
   for text_len, text_table in pairs(texts_by_num_items_idx) do
      for num_items, mat in pairs(text_table) do
         local curr_num_items = num_items
         if num_items == 0 then
            curr_num_items = 1
         end
         local curr_text_idx = 1
         local curr_text_idx_dup = 1
         local num_texts_of_len = texts_by_num_items[text_len][num_items]
         local randomized_set = nil
         -- Since we removed leftovers earlier, the number of texts in the current set will divide evenly into batches.
         local num_batches_of_lengths = texts_by_num_items[text_len][num_items] / batch_size

         -- Update max text length and max num ingredients for this current set of batches
         if num_texts_of_len ~= 0 then
            if self.max_text_length == nil then
               self.max_text_length = text_len
            elseif self.max_text_length < text_len then
               self.max_text_length = text_len
            end
            if self.max_num_items == nil then
               self.max_num_items = curr_num_items
            elseif self.max_num_items < curr_num_items + 1 then
               self.max_num_items = curr_num_items
            end
            randomized_set = torch.range(1,texts_by_num_items[text_len][num_items])
         end

         -- Loop through the batches and for each batch, find its max sizes for
         -- goal length and ingredient length (in tokens).
         -- The max sizes will be used to generate properly-sized tensors to store
         -- the batch information.
         for b=1, num_batches_of_lengths do
            local max_goal_length = 0
            local max_item_length = 0
            local curr_batch_len = torch.zeros(4)
            for r=1,batch_size do
               local random_text_idx = randomized_set[curr_text_idx_dup]
               local true_text_idx = texts_by_num_items_idx[text_len][num_items][random_text_idx]
               local text_offset_idx = text_to_pos[text_len][true_text_idx]
               local text_offsets = offsets[text_offset_idx]
               local goal_len = text_offsets[3]
               if max_goal_length < goal_len then
                  max_goal_length = goal_len
               end
               local itemset_index = text_offsets[6]
               local itemset = items[num_items][itemset_index]
               for i=1,num_items do
                  local item_length = itemset[i][1]
                  if max_item_length < item_length then
                     max_item_length = item_length
                  end
               end
               curr_text_idx_dup = curr_text_idx_dup + 1
            end
            curr_batch_len[1] = text_len
            curr_batch_len[2] = max_goal_length
            curr_batch_len[3] = curr_num_items
            curr_batch_len[4] = max_item_length

            if self.max_item_length == nil then
               self.max_item_length = max_item_length
            elseif self.max_item_length < max_item_length then
               self.max_item_length = max_item_length
            end
            if self.max_goal_length == nil then
               self.max_goal_length = max_goal_length
            elseif self.max_goal_length < max_goal_length then
               self.max_goal_length = max_goal_length
            end

            ---------------------------------------------------------
            -- Initialize data structures.
            ---------------------------------------------------------
            local text_mat = torch.zeros(text_len, batch_size)
            local goal_mat = torch.zeros(batch_size, max_goal_length)
            local items_mat = torch.zeros(batch_size, curr_num_items, max_item_length)
            local ref_type_mat = torch.zeros(text_len, batch_size, 3):float()
            local true_new_item_atten_mat = torch.zeros(text_len, batch_size, curr_num_items):float()
            local true_used_item_atten_mat = torch.zeros(text_len, batch_size, curr_num_items):float()
            -----------------------------------------------------------

            -- Fill in information for each batch.
            for r=1,batch_size do
               local random_text_idx = randomized_set[curr_text_idx]
               local true_text_idx = texts_by_num_items_idx[text_len][num_items][random_text_idx]
               local text = texts[text_len][true_text_idx]
               local text_offset_idx = text_to_pos[text_len][true_text_idx]
               local text_offsets = offsets[text_offset_idx]
               if data_type == 'dev' then -- dev batches don't need to be randomized
                  self.dev_random_batch_ordering[text_offset_idx] = batch_counter
               end
               for i=1,text_len do
                  local token = text[i][1]
                  text_mat[i][r] = token
                  local nonitem_prob = text[i][2]
                  local used_item_prob = text[i][3]
                  local new_item_prob = text[i][4]
                  
                  if nonitem_prob == 0 then
                     ref_type_mat[i][r][1] = 1
                     ref_type_mat[i][r][2] = 0
                     ref_type_mat[i][r][3] = 0
                  elseif used_item_prob > 0 then
                     ref_type_mat[i][r][1] = 0
                     ref_type_mat[i][r][2] = 1
                     ref_type_mat[i][r][3] = 0
                     true_used_item_atten_mat[i][r][used_item_prob] = 1
                  elseif used_item_prob == -1 then
                     -- all ingredients, but not using, so call a nonfood
                     ref_type_mat[i][r][1] = 1
                     ref_type_mat[i][r][2] = 0
                     ref_type_mat[i][r][3] = 0
                  else
                     if new_item_prob == -1 then
                        -- all ingredients, but not using, so call a nonfood
                        ref_type_mat[i][r][1] = 1
                        ref_type_mat[i][r][2] = 0
                        ref_type_mat[i][r][3] = 0
                     else
                        ref_type_mat[i][r][1] = 0
                        ref_type_mat[i][r][2] = 0
                        ref_type_mat[i][r][3] = 1
                        true_new_item_atten_mat[i][r][new_item_prob] = 1
                     end
                  end 
               end
               local goal_length = text_offsets[3]
               local tbin = text_offsets[4]
               local goal = goals[goal_length][tbin]
               for i=1,goal_length do
                  goal_mat[r][i] = goal[i]
               end
               if goal_length == 0 then
                  goal_mat[r][1] = goal_unk
               end
               local items_number = text_offsets[5]
               local isbin = text_offsets[6]
               local item_info = items[items_number][isbin]
               if num_items == 0 then
                  items_mat[r][1][1] = item_unk
               end
               for i=1,items_number do
                  local item_length = item_info[i][1]
                  local ibin = item_info[i][2]
                  for j=1,item_length do
                     items_mat[r][i][j] = item[item_length][ibin][j]
                  end
               end

               curr_text_idx = curr_text_idx + 1
            end

            -- Add batch to table of batches
            if opt.cpu then
               table.insert(text_batches, text_mat:contiguous():float())
               table.insert(goal_batches, goal_mat:contiguous():float())
               table.insert(items_batches, items_mat:contiguous():float())
               table.insert(batch_len, curr_batch_len:contiguous():float())
               table.insert(ref_type_batches, ref_type_mat:contiguous():float())
               table.insert(true_new_item_atten_batches, true_new_item_atten_mat:contiguous():float())
               table.insert(true_used_item_atten_batches, true_used_item_atten_mat:contiguous():float())
            else
               table.insert(text_batches, text_mat:contiguous():float():cuda())
               table.insert(goal_batches, goal_mat:contiguous():float():cuda())
               table.insert(items_batches, items_mat:contiguous():float():cuda())
               table.insert(batch_len, curr_batch_len:contiguous():float():cuda())
               table.insert(ref_type_batches, ref_type_mat:contiguous():float():cuda())
               table.insert(true_new_item_atten_batches, true_new_item_atten_mat:contiguous():float():cuda())
               table.insert(true_used_item_atten_batches, true_used_item_atten_mat:contiguous():float():cuda())
            end

            if self.max_num_words < text_len then
               self.max_num_words = text_len
            end
            batch_counter = batch_counter + 1
         end
      end
   end

   -- creatitem batch info labels dependitem on if this is 'train' or 'dev'
   if data_type == 'train' then
      self.text_batches = text_batches
      self.batch_len = batch_len
      self.goal_batches = goal_batches
      self.items_batches = items_batches
      self.ref_type_batches = ref_type_batches
      self.true_new_item_atten_batches = true_new_item_atten_batches
      self.true_used_item_atten_batches = true_used_item_atten_batches
   elseif data_type == 'dev' then
      self.dev_text_batches = text_batches
      self.dev_batch_len = batch_len
      self.dev_goal_batches = goal_batches
      self.dev_items_batches = items_batches
      self.dev_ref_type_batches = ref_type_batches
      self.dev_true_new_item_atten_batches = true_new_item_atten_batches
      self.dev_true_used_item_atten_batches = true_used_item_atten_batches
   end
end

-----------------------------------------------------------------------------------------------------------
-- Create data structures for a given train-dev information.
-----------------------------------------------------------------------------------------------------------
-- ***If you want to use this for a test set, set the dev_data_dir to the test set directory.***
-- The dev set only means you have minibatches of 1 and they aren't randomized.
-----------------------------------------------------------------------------------------------------------
-- train_data_dir: trainitem data directory, also directory that holds the dictionary torch files
-- dev_data_dir (optional): dev or test set directory.
-- batch_size: batch size for trainitem data
-- predict (boolean): whether or not we are just generatitem new texts. If true, we don't load the trainitem
--    data, but we DO still load the dictionaries from the trainitem data directory
-- info: data file version information
-- opt: flag information
-------------------------------------------------------------------------------------------------------------
function RecipeDataMinibatchLoader.create(train_data_dir, dev_data_dir, batch_size, predict, info, opt, randomize_train)

   local self = {}
   setmetatable(self, RecipeDataMinibatchLoader)

   -- load dictionaries
   self.max_num_words = 0
   local dict_file = path.join(train_data_dir, info .. 'dict.torch')
   local dict = torch.load(dict_file)
   local item_dict_file = path.join(train_data_dir, info .. 'itemdict.torch')
   local item_dict = torch.load(item_dict_file)
   local goal_dict_file = path.join(train_data_dir, info .. 'goaldict.torch')
   local goal_dict = torch.load(goal_dict_file)

   -- count vocabularies
   self.dict = dict
   self.item_dict = item_dict
   self.goal_dict = goal_dict
   self.vocab_size = 0
   for _ in pairs(self.dict.index_to_symbol) do
      self.vocab_size = self.vocab_size + 1
   end
   self.item_vocab_size = 0
   for _ in pairs(self.item_dict.index_to_symbol) do
      self.item_vocab_size = self.item_vocab_size + 1
   end
   self.goal_vocab_size = 0
   for _ in pairs(self.goal_dict.index_to_symbol) do
      self.goal_vocab_size = self.goal_vocab_size + 1
   end
   self.vocab_size = self.vocab_size + 1
      dict.symbol_to_index['<ALL>'] = self.vocab_size
      dict.index_to_symbol[self.vocab_size] = '<ALL>'

   self.pad = 0

   -- settitem batch size
   self.batch_size = batch_size
   self.num_batches_to_sort_at_once = num_batches_to_sort_at_once

   self.split_sizes = {0, 0, 0}

   -- Load trainitem data if provided and we want to use it.
   if train_data_dir ~= nil and not predict then
      self:loadData('train', train_data_dir, dict, item_dict, goal_dict, randomize_train, info, opt)
   end

   -- Load dev data if provided.
   if dev_data_dir ~= nil then
      self:loadData('dev', dev_data_dir, dict, item_dict, goal_dict, false, info, opt)
   end
   self.batch_idx = {0, 0, 0}
   print('max num words: ' .. self.max_num_words)
   print('vocab_size: ' .. self.vocab_size)
   print('item vocab_size: ' .. self.item_vocab_size)
   print('goal vocab_size: ' .. self.goal_vocab_size)

   collectgarbage()
   return self
end

--------------------------------------------------------------------------------
-- Reset batch index back to 0 if we come to the end of the batch.
--------------------------------------------------------------------------------
function RecipeDataMinibatchLoader:reset_batch_pointer(split_index, batch_index)
   batch_index = batch_index or 0
   self.batch_idx[split_index] = batch_index
end

-------------------------------------------------------
-- Loads a particular batch
-------------------------------------------------------
function RecipeDataMinibatchLoader:get_non_randomized_training_batch(batch_index)
   return self.text_batches[batch_index], self.goal_batches[batch_index], self.items_batches[batch_index], self.batch_len[batch_index], self.ref_type_batches[batch_index], self.true_new_item_atten_batches[batch_index], self.true_used_item_atten_batches[batch_index]
end


----------------------------------------------------------
-- Load next batch.
----------------------------------------------------------
--
-- split_index: 1 for train, 2 for dev
----------------------------------------------------------
function RecipeDataMinibatchLoader:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwitem somethitem up
        local split_names = {'train', 'val'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val
    self.batch_idx[split_index] = self.batch_idx[split_index] + 1
    if self.batch_idx[split_index] > self.split_sizes[split_index] then
        self.batch_idx[split_index] = 1 -- cycle around to beginnitem
    end
    -- pull out the correct next batch
   if split_index == 1 then
      local ix = self.random_batch_ordering[self.batch_idx[split_index]]
      return self.text_batches[ix], self.goal_batches[ix], self.items_batches[ix], self.batch_len[ix], self.ref_type_batches[ix], self.true_new_item_atten_batches[ix], self.true_used_item_atten_batches[ix]
   end
   if split_index == 2 then 
      local ix = self.dev_random_batch_ordering[self.batch_idx[split_index]]
      return self.dev_text_batches[ix], self.dev_goal_batches[ix], self.dev_items_batches[ix], self.dev_batch_len[ix], self.dev_ref_type_batches[ix], self.dev_true_new_item_atten_batches[ix], self.dev_true_used_item_atten_batches[ix]
   end
end

return RecipeDataMinibatchLoader



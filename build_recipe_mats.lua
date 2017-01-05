require('torch')

local stringx = require('pl.stringx')

torch.setdefaulttensortype('torch.LongTensor')

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Build torch serialized version of recipes.')
cmd:text()

cmd:option('-inDirectory', '', 'The input recipe directory')
cmd:option('-useDictionary', '', 'Dictionary file')
cmd:option('-listFilename', '', 'The list of recipe filenames in the input directory')
cmd:option('-outDirectory', '', 'The output directory')
cmd:option('-dictionaryCutoff', 5, 'Cutoff for number of times a word must be seen.')
cmd:option('-info', 'emnlp', 'Versioning information')
cmd:option('-devDir', '', 'Dev directory, to ensure no dev titles are used when building training mats')
cmd:option('-testDir', '', 'Test directory, to ensure no test titles are used when building train/dev mats')

opt = cmd:parse(arg)

-----------------------------------------------------
-- Add a word to a dictionary.
-----------------------------------------------------
local function add_to_dictionary(dict, word_id, word)
   if dict.symbol_to_index[word] == nil then
      word_id = word_id + 1
      dict.symbol_to_index[word] = word_id
      dict.index_to_symbol[word_id] = word
   end
   return word_id
end

---------------------------------------------------
-- Count numbers of recipes of different lengths in
-- order to create tensors of the proper size.
---------------------------------------------------
local function count_recipe_lengths(counter)
   local nrecipe = 1
   counter.cantuse = {}
   counter.cantusefull = {}

   -- Read in dev directory titles in provided.
   -- DO NOT provide for generating the dev set.
   local cantuse_title_tokens = {}
   if opt.devDir ~= '' then
   local filelist = io.open(opt.listFilename, 'r')
      for recipefilename in filelist:lines() do
         local recipefile = io.open(opt.devDir .. recipefilename, 'r')
         for line in recipefile:lines() do
            if stringx.startswith(line, "Title:") or stringx.startswith(line, "title:") then
               local title = string.sub(line, 8, -1)
               local orig_title_tokens = stringx.split(stringx.strip(title:lower()))

               local new_tokens = {}
               for i=1,#orig_title_tokens do
                  local token = orig_title_tokens[i]
                  if token:find("[%a]") ~= nil and token:find("'s") == nil and token ~= 'and' and token ~= 'with' then
                     table.insert(new_tokens, token)
                  end
               end
               if #new_tokens ~= 0 then
                  table.sort(new_tokens, function(a, b) return a < b end)
                  local new_title_string = table.concat(new_tokens, " ")
                  counter.cantuse[new_title_string] = true
                  local tokens_by_name = {}
                  for i=1,#new_tokens do
                     tokens_by_name[new_tokens[i] ] = true
                  end
                  table.insert(cantuse_title_tokens, tokens_by_name)
               end
            end
         end
      end
   end
   -- Read in test directory titles in provided.
   -- DO NOT provide for generating the test set.
   if opt.testDir ~= '' then
      local filelist = io.open(opt.listFilename, 'r')
      for recipefilename in filelist:lines() do
         local recipefile = io.open(opt.testDir .. recipefilename, 'r')
         for line in recipefile:lines() do
            if stringx.startswith(line, "Title:") or stringx.startswith(line, "title:") then
               local title = string.sub(line, 8, -1)
               -- Preprocess title to remove certain characters.
               local orig_title_tokens = stringx.split(stringx.strip(title:lower()))
   
               local new_tokens = {}
               for i=1,#orig_title_tokens do
                  local token = orig_title_tokens[i]
                  if token:find("[%a]") ~= nil and token:find("'s") == nil and token ~= 'and' and token ~= 'with' then
                     table.insert(new_tokens, token)
                  end
               end
               if #new_tokens ~= 0 then
                  table.sort(new_tokens, function(a, b) return a < b end)
                  local new_title_string = table.concat(new_tokens, " ")
                  counter.cantuse[new_title_string] = true
                  local tokens_by_name = {}
                  for i=1,#new_tokens do
                     tokens_by_name[new_tokens[i] ] = true
                  end
                  table.insert(cantuse_title_tokens, tokens_by_name)
               end
            end
         end
      end
   end
   local filelist = io.open(opt.listFilename, 'r')
   for recipefilename in filelist:lines() do
      print(recipefilename)
      local recipefile = io.open(opt.inDirectory .. recipefilename, 'r')
      local sent = nil
      local seen_pred = false
      local step_table = {}
      local recipe_len = 1
      local title_tokens = {}
      local ing_strings = nil
      local switch_evidence = false
      local title = nil
      local original_title = nil
      for line in recipefile:lines() do
         if stringx.startswith(line, "Title:") or stringx.startswith(line, "title:") then
            title = string.sub(line, 8, -1)
            original_title = title
            -- Preprocess title to remove certain characters.
            local orig_title_tokens = stringx.split(stringx.strip(title:lower()))

            for i=1,#orig_title_tokens do
               local token = orig_title_tokens[i]
               if token:find("[%a]") ~= nil and token:find("'s") == nil and token ~= 'and' and token ~= 'with' then
                  table.insert(title_tokens, token)
               end
            end
            if #title_tokens ~= 0 then
               table.sort(title_tokens, function(a, b) return a < b end)
               local new_title_string = table.concat(title_tokens, " ")
               if counter.cantuse[new_title_string] == true then
                  counter.cantusefull[original_title] = true
                  title = nil
               else
                  local too_much_overlap = false
                  for c=1,#cantuse_title_tokens do
                     local bad_tokens = cantuse_title_tokens[c]
                     local overlap = 0
                     for i=1,#title_tokens do
                        if bad_tokens[title_tokens[i] ] == true then
                           overlap = overlap + 1
                        end
                        if overlap >= 3 then
                           too_much_overlap = true
                           break
                        end
                     end
                     if too_much_overlap then
                        break
                     end
                  end
                  if too_much_overlap then
                     counter.cantusefull[original_title] = true
                     title = nil
                  else
                  end
                  if title ~= nil then
                     for i=1,#title_tokens do
                        local token = title_tokens[i]
                        counter.title_counts[token] = (counter.title_counts[token] or 0) + 1   
                     end
                  end
               end
            end
         elseif stringx.startswith(line, "Categories:") or stringx.startswith(line, "categories:") then
         elseif stringx.startswith(line, "Servings:") or stringx.startswith(line, "servings:") then
         elseif stringx.strip(line) == "" then
         elseif stringx.startswith(line, "Ingredients:") and title ~= nil then
            local ingline = stringx.strip(string.sub(line, 14, -1)):lower()
            local origingline = ingline
            -- Preprocess ingredients to remove certain characters.
            ingline = string.gsub(ingline, ",", "")
            ingline = string.gsub(ingline, ";", "")
            ingline = string.gsub(ingline, "%^", "")
            ingline = string.gsub(ingline, "%*", "")
            ingline = string.gsub(ingline, "%.", "")
            ingline = string.gsub(ingline, "%&", "")
            ingline = string.gsub(ingline, ":", "")
            ingline = string.gsub(ingline, "<", "")
            ingline = string.gsub(ingline, ">", "")
            ingline = string.gsub(ingline, "~", "")
            ingline = string.gsub(ingline, "=", "")
            ingline = string.gsub(ingline, "+", "")
            ingline = string.gsub(ingline, "%(", "")
            ingline = string.gsub(ingline, "%)", "")
            ingline = string.gsub(ingline, " %- ", " ")
            ingline = string.gsub(ingline, "% or% ", " ")
            ingline = string.gsub(ingline, "%-lrb%-", "")
            ingline = string.gsub(ingline, "%-LRB%-", "")
            ingline = string.gsub(ingline, "%-rrb%-", "")
            ingline = string.gsub(ingline, "%-RRB%-", "")
            ingline = string.gsub(ingline, "  ", " ")
            ingline = string.gsub(ingline, "  ", " ")
            ingline = string.gsub(ingline, "  ", " ")
            ing_strings = stringx.split(ingline, '\t')
         elseif stringx.startswith(line, "END") then
            if title ~= nil and #step_table ~= 0 then
               counter.nrecipes = counter.nrecipes + 1
               for i=1,#ing_strings do
                  local ing = ing_strings[i]
                  local ing_tokens = stringx.split(ing)
                  local new_length = 0
                  local new_tokens = {}
                  for j=1,#ing_tokens do
                     local token = ing_tokens[j]
                     -- Only use word tokens in the ingredient and remove amount information.
                     -- 2 character tokens below are MealMaster amount codes
                     if token:find("[%a]") ~= nil then
                        if token ~= "x" and token ~= "sm" and token ~= "md" and token ~= "lg" and token ~= "cn" and token ~= "pk"
                           and token ~= "pn" and token ~= "dr" and token ~= "ds" and token ~= "ct" and token ~= "bn" and token ~= "sl"
                           and token ~= "ea" and token ~= "t" and token ~= "ts" and token ~= "T" and token ~= "tb" and token ~= "fl"
                           and token ~= "c" and token ~= "pt" and token ~= "qt" and token ~= "ga" and token ~= "oz" and token ~= "lb"
                           and token ~= "ml" and token ~= "cb" and token ~= "cl" and token ~= "dl" and token ~= "l" and token ~= "mg"
                           and token ~= "cg" and token ~= "dg" and token ~= "g" and token ~= "kg" then
                           counter.ing_counts[token] = (counter.ing_counts[token] or 0) + 1
                           table.insert(new_tokens, token)
                           new_length = new_length + 1
                        end
                     end
                  end
                  if #new_tokens ~= 0 then
                     table.sort(new_tokens, function(a, b) return a < b end)
                     local new_ing_string = table.concat(new_tokens, " ")
                     counter.ing_length_counts[#new_tokens] = (counter.ing_length_counts[#new_tokens] or 0) + 1
                     counter.lost_map[ing] = new_ing_string
                  else
                     counter.ing_length_counts[1] = (counter.ing_length_counts[1] or 0) + 1
                     counter.lost_map[ing] = "<unk>"
                  end 
               end
               local max_step_len = 0
               for s=1,#step_table do
                  local step = step_table[s]
                  -- add 1 for </s> or </text>
                  local step_tokens = stringx.split(step)
                  local clean_token_count = 0
                  for t=1,#step_tokens do
                     local token = step_tokens[t]
                     if token ~= '' then
                        counter.word_counts[token] = (counter.word_counts[token] or 0) + 1
                        clean_token_count = clean_token_count + 1
                     end
                  end
                  recipe_len = recipe_len + clean_token_count + 1
               end
               counter.num_steps_counts[#step_table] = (counter.num_steps_counts[#step_table] or 0) + 1
               counter.full_recipe_len_counts[recipe_len] = (counter.full_recipe_len_counts[recipe_len] or 0) + 1
               counter.title_length_counts[#title_tokens] = (counter.title_length_counts[#title_tokens] or 0) + 1
               counter.ing_number_counts[#ing_strings] = (counter.ing_number_counts[#ing_strings] or 0) + 1

               nrecipe = nrecipe + 1
            end
            title = nil
            sent = nil
            original_title = nil
            seen_pred = false
            step_table = {}
            title_tokens = {}
            ing_strings = nil
            recipe_len = 1
            switch_evidence = false
         elseif switch_evidence then
            switch_evidence = false
         else
            sent = line:lower()
            switch_evidence = true
            table.insert(step_table, stringx.strip(sent))
         end
	   end
   end
   return counter
end

local function create_dictionary(counter)
   local dict = {symbol_to_index = {},
                 index_to_symbol = {}}
   local title_dict = {symbol_to_index = {},
                 index_to_symbol = {}}
   local ing_dict = {symbol_to_index = {},
                 index_to_symbol = {}}
   local word_id = 0
   word_id = add_to_dictionary(dict, word_id, '<unk>')
   word_id = add_to_dictionary(dict, word_id, '</text>')
   word_id = add_to_dictionary(dict, word_id, '<text>')
   word_id = add_to_dictionary(dict, word_id, '\n')
   for word,count in pairs(counter.word_counts) do
      if count >= opt.dictionaryCutoff then
         word_id = add_to_dictionary(dict, word_id, word)
      end
   end
   print('dictionary size: ' .. word_id)
   counter.word_counts = {}
   local title_id = 0
   title_id = add_to_dictionary(title_dict, title_id, '<unk>')
   for word,count in pairs(counter.title_counts) do
      if count >= opt.dictionaryCutoff then
         title_id = add_to_dictionary(title_dict, title_id, word)
      end
   end
   print('title dictionary size: ' .. title_id)
   counter.title_counts = {}

   local ing_id = 0
   ing_id = add_to_dictionary(ing_dict, ing_id, '<unk>')
   for word,count in pairs(counter.ing_counts) do
      if count >= opt.dictionaryCutoff then
         ing_id = add_to_dictionary(ing_dict, ing_id, word)
      end
   end
   counter.ing_counts = {}
   return dict, ing_dict, title_dict
end

local function add_to_dictionaries(counter, dict, ing_dict, title_dict)
   local word_id = 0
   for _ in pairs(dict.index_to_symbol) do
      word_id = word_id + 1
   end
   for word,count in pairs(counter.word_counts) do
      if count >= opt.dictionaryCutoff then
         word_id = add_to_dictionary(dict, word_id, word)
      end
   end
   print('dictionary size: ' .. word_id)
   counter.word_counts = {}
   local title_id = 0
   for _ in pairs(title_dict.index_to_symbol) do
      title_id = title_id + 1
   end
   for word,count in pairs(counter.title_counts) do
      if count >= opt.dictionaryCutoff then
         title_id = add_to_dictionary(title_dict, title_id, word)
      end
   end
   print('title dictionary size: ' .. title_id)
   counter.title_counts = {}

   local ing_id = 0
   for _ in pairs(ing_dict.index_to_symbol) do
      ing_id = ing_id + 1
   end
   for word,count in pairs(counter.ing_counts) do
      if count >= opt.dictionaryCutoff then
         ing_id = add_to_dictionary(ing_dict, ing_id, word)
      end
   end
   counter.ing_counts = {}
   return dict, ing_dict, title_dict
end

local function build_recipe_matrix(counter, dict, ing_dict, title_dict)
   local full_recipe_to_pos = {}
   local recipe_mat = {}
   local of_recipe_len = {}

   for length,count in pairs(counter.full_recipe_len_counts) do
      recipe_mat[length] = torch.zeros(count, length, 4):float()
      full_recipe_to_pos[length] = torch.zeros(count):long()
      of_recipe_len[length] = 1
   end

   local title_mat = {}
   local true_title_mat = {}
   local of_title_len = {}
   for length,count in pairs(counter.title_length_counts) do
      title_mat[length] = torch.zeros(count, length):long()
      of_title_len[length] = 1
   end

   local ing_mat = {}
   local of_ing_len = {}
   for length,count in pairs(counter.ing_length_counts) do
      ing_mat[length] = torch.zeros(count, length):long()
      of_ing_len[length] = 1
   end

   local ings_mat = {}
   local true_ings_mat = {}
   local of_ing_num = {}
   for length,count in pairs(counter.ing_number_counts) do
      ings_mat[length] = torch.zeros(count, length, 2):long()
      of_ing_num[length] = 1
   end

   local pos = torch.zeros(counter.nrecipes, 6):long()

   local unk_index = dict.symbol_to_index['<unk>']
   local ing_unk_index = ing_dict.symbol_to_index['<unk>']
   local title_unk_index = title_dict.symbol_to_index['<unk>']
   local start_index = dict.symbol_to_index['<text>']
   local end_index = dict.symbol_to_index['</text>']
   local line_break_index = dict.symbol_to_index['\n']

   local nrecipe = 1

   local filelist = io.open(opt.listFilename, 'r')
   for recipefilename in filelist:lines() do
      print(recipefilename)
      local recipefile = io.open(opt.inDirectory .. recipefilename, 'r')
      local sent = nil
      local title_tokens = {}
      local ing_strings = nil
      local seen_pred = false
      local step_table = {}
      local recipe_len = 1
      local prob_str_table = {}
      local title = nil
      local switch_prob = false
      for line in recipefile:lines() do
         if stringx.startswith(line, "Title:") or stringx.startswith(line, "title:") then
            title = string.sub(line, 8, -1)
            if counter.cantusefull[title] == nil then
               local orig_title_tokens = stringx.split(stringx.strip(title:lower()))
         
               for i=1,#orig_title_tokens do
                  local token = orig_title_tokens[i]
                  if token:find("[%a]") ~= nil and token:find("'s") == nil and token ~= 'and' and token ~= 'with' then
                     table.insert(title_tokens, token)
                  end
               end
            else
               title = nil
            end
         elseif stringx.startswith(line, "Categories:") or stringx.startswith(line, "categories:") then
         elseif stringx.startswith(line, "Servings:") or stringx.startswith(line, "servings:") then
         elseif stringx.startswith(line, "Ingredients:") then
            local ingline = stringx.strip(string.sub(line, 14, -1)):lower()
            ingline = string.gsub(ingline, ",", "")
            ingline = string.gsub(ingline, ";", "")
            ingline = string.gsub(ingline, "%^", "")
            ingline = string.gsub(ingline, "%*", "")
            ingline = string.gsub(ingline, "%.", "")
            ingline = string.gsub(ingline, "%&", "")
            ingline = string.gsub(ingline, ":", "")
            ingline = string.gsub(ingline, "<", "")
            ingline = string.gsub(ingline, ">", "")
            ingline = string.gsub(ingline, "~", "")
            ingline = string.gsub(ingline, "=", "")
            ingline = string.gsub(ingline, "+", "")
            ingline = string.gsub(ingline, "%(", "")
            ingline = string.gsub(ingline, "%)", "")
            ingline = string.gsub(ingline, " %- ", " ")
            ingline = string.gsub(ingline, " or ", " ")
            ingline = string.gsub(ingline, "%-lrb%-", "")
            ingline = string.gsub(ingline, "%-LRB%-", "")
            ingline = string.gsub(ingline, "%-rrb%-", "")
            ingline = string.gsub(ingline, "%-RRB%-", "")
            ingline = string.gsub(ingline, "  ", " ")
            ingline = string.gsub(ingline, "  ", " ")
            ingline = string.gsub(ingline, "  ", " ")
            ing_strings = stringx.split(ingline, '\t')
        elseif stringx.strip(line) == "" then
        elseif stringx.startswith(line, "END") then
         
         if title ~= nil and #step_table ~= 0 then       
            for i=1,#step_table do
               local step = step_table[i]
               local step_tokens = stringx.split(step)
               for p=1,#step_tokens do
                  local token = step_tokens[p]
                  if token ~= '' then
                     recipe_len = recipe_len + 1
                  end
               end
               recipe_len = recipe_len + 1
            end

            local nbin = of_recipe_len[recipe_len]
            pos[nrecipe][1] = recipe_len
            pos[nrecipe][2] = nbin
            of_recipe_len[recipe_len] = nbin + 1

            recipe_mat[recipe_len][nbin][1][1] = start_index 
            recipe_mat[recipe_len][nbin][1][2] = 0.0
            recipe_mat[recipe_len][nbin][1][3] = 0.0
            recipe_mat[recipe_len][nbin][1][4] = 0.0
            full_recipe_to_pos[recipe_len][nbin] = nrecipe

            local tbin = of_title_len[#title_tokens]
            of_title_len[#title_tokens] = tbin + 1
            local cube = false
            for i=1,#title_tokens do
               local token = title_tokens[i]
               if token == 'cube' then
                  cube = true
               end
            end
            for i=1,#title_tokens do
               local token = title_tokens[i]
               local tindex = title_dict.symbol_to_index[token] or 0
               title_mat[#title_tokens][tbin][i] = tindex
            end
            local full_title = table.concat(title_tokens, " ")
            true_title_mat[nrecipe] = full_title
            pos[nrecipe][3] = #title_tokens
            pos[nrecipe][4] = tbin

            local isbin = of_ing_num[#ing_strings]
            of_ing_num[#ing_strings] = isbin + 1
            local all_ings = table.concat(ing_strings, '\t')
            true_ings_mat[nrecipe] = all_ings
            for i=1,#ing_strings do
               local ing = ing_strings[i]
               local new_ing = counter.lost_map[ing]
            --   print(new_ing)
               local tokens = stringx.split(new_ing, " ")
               local ibin = of_ing_len[#tokens]
               of_ing_len[#tokens] = ibin + 1
               ings_mat[#ing_strings][isbin][i][1] = #tokens
               ings_mat[#ing_strings][isbin][i][2] = ibin
               for j,token in ipairs(tokens) do
                  local tindex = ing_dict.symbol_to_index[token] or ing_unk_index
                  ing_mat[#tokens][ibin][j] = tindex
               end
            end
            pos[nrecipe][5] = #ing_strings
            pos[nrecipe][6] = isbin

            local index = 2
            for i=1,#step_table do
               local step = step_table[i]
               step = string.gsub(step, " = ", " ")
               step = string.gsub(step, " ~ ", " ")
               step = string.gsub(step, " + ", " ")
               step = string.gsub(step, "=", "")
               step = string.gsub(step, "~", "")
               step = string.gsub(step, "+", "")
               step = string.gsub(step, "  ", " ")
               step = string.gsub(step, "  ", " ")
               step = string.gsub(step, "  ", " ")
               step = string.gsub(step, "  ", " ")
               local evid_strs = prob_str_table[i]
               local step_tokens = stringx.split(step)
               local evidence = stringx.split(evid_strs)
               if i ~= 1 then 
                  recipe_mat[recipe_len][nbin][index][1] = line_break_index 
                  recipe_mat[recipe_len][nbin][index][2] = 0.0
                  recipe_mat[recipe_len][nbin][index][3] = 0.0
                  recipe_mat[recipe_len][nbin][index][4] = 0.0
                  index = index + 1
               end
               for j=1,#step_tokens do
                  local token = step_tokens[j]
                  if token ~= '=' and token ~= '~' and token ~= '+' then
                     token = string.gsub(token, "=", "")
                     token = string.gsub(token, "+", "")
                     token = string.gsub(token, "~", "")
                     if token ~= '' then
                        local tindex = dict.symbol_to_index[token] or unk_index
                        local evid_split = stringx.split(evidence[j], '_')
                        recipe_mat[recipe_len][nbin][index][1] = tindex
                        if tindex == unk_index then
                           recipe_mat[recipe_len][nbin][index][2] = 0.0
                           recipe_mat[recipe_len][nbin][index][3] = 0.0
                           recipe_mat[recipe_len][nbin][index][4] = 0.0
                        elseif tonumber(evid_split[2]) == -1 or tonumber(evid_split[3]) == -1 then
                           -- -1 represents "all ingredients" token. For now, this is ignored.
                           recipe_mat[recipe_len][nbin][index][2] = 0.0
                           recipe_mat[recipe_len][nbin][index][3] = 0.0
                           recipe_mat[recipe_len][nbin][index][4] = 0.0
                        else
                           recipe_mat[recipe_len][nbin][index][2] = tonumber(evid_split[1])
                           recipe_mat[recipe_len][nbin][index][3] = tonumber(evid_split[2])
                           recipe_mat[recipe_len][nbin][index][4] = tonumber(evid_split[3])
                        end
                        index = index + 1
                     end
                  end
               end
            end
            recipe_mat[recipe_len][nbin][index][1] = end_index 
            recipe_mat[recipe_len][nbin][index][2] = 0.0
            recipe_mat[recipe_len][nbin][index][3] = 0.0
            recipe_mat[recipe_len][nbin][index][4] = 0.0
            nrecipe = nrecipe + 1
            end
            title = nil
            sent = nil
            title_tokens = {}
            ing_strings = nil
            seen_pred = false
            recipe_len = 1
            step_table = {}
            prob_str_table = {}
            switch_prob = false
         elseif switch_prob then
            table.insert(prob_str_table, line)
            switch_prob = false
         else
            sent = line:lower()
            switch_prob = true
            table.insert(step_table, stringx.strip(sent))
         end
      end
   end
   print(nrecipe)
   return recipe_mat, full_recipe_to_pos, title_mat, true_title_mat, ings_mat, true_ings_mat, ing_mat, pos
end

local function main()
   local recipe_counter = {
      nrecipes = 0,
      num_steps_counts = {},
      full_recipe_len_counts = {},
      title_length_counts = {},
      ing_number_counts = {},
      ing_length_counts = {},
      word_counts = {},
      ing_counts = {},
      title_counts = {},
      lost_map = {}}
   local dict = {}
   local ing_dict = {}
   local title_dict = {}
   if opt.useDictionary ~= '' then
      print('Using dictionary ' .. opt.useDictionary)
      dict = torch.load(opt.useDictionary .. '.dict.torch');
      ing_dict = torch.load(opt.useDictionary .. '.itemdict.torch');
      title_dict = torch.load(opt.useDictionary .. '.goaldict.torch');
      recipe_counter.lost_map = dict.lost_map
   end
   print('Counting recipe lengths...')
   recipe_counter = count_recipe_lengths(recipe_counter)
   if opt.useDictionary == '' then
      dict, ing_dict, title_dict = create_dictionary(recipe_counter)
      dict.ing_counts = recipe_counter.ing_counts
      dict.lost_map = recipe_counter.lost_map
      dict.word_counts = recipe_counter.word_counts
   elseif opt.dictionaryCutoff == 0 then
      dict, ing_dict, title_dict = add_to_dictionaries(recipe_counter, dict, ing_dict, title_dict)
   end

   print('Building recipe matrices...')
   local recipe_mat, recipe_to_pos, title_mat, true_title_mat, ings_mat, true_ings_mat, ing_mat, pos = build_recipe_matrix(recipe_counter, dict, ing_dict, title_dict)
   if opt.useDictionary ~= '' then
      dict.ing_counts = nil
      dict.lost_map = nil
      dict.word_counts = nil
   end
   torch.save(opt.outDirectory .. opt.info .. '.dict.torch', dict)
   torch.save(opt.outDirectory .. opt.info .. '.itemdict.torch', ing_dict)
   torch.save(opt.outDirectory .. opt.info .. '.goaldict.torch', title_dict)
   torch.save(opt.outDirectory .. 'text.' .. opt.info .. '.mat.torch', recipe_mat)
   torch.save(opt.outDirectory .. 'text_to_pos.' .. opt.info .. '.mat.torch', recipe_to_pos)
   torch.save(opt.outDirectory .. 'goal.' .. opt.info .. '.mat.torch', title_mat)
   torch.save(opt.outDirectory .. 'true_goal.' .. opt.info .. '.mat.torch', true_title_mat)
   torch.save(opt.outDirectory .. 'items.' .. opt.info .. '.mat.torch', ings_mat)
   torch.save(opt.outDirectory .. 'true_items.' .. opt.info .. '.mat.torch', true_ings_mat)
   torch.save(opt.outDirectory .. 'item.' .. opt.info .. '.mat.torch', ing_mat)
   torch.save(opt.outDirectory .. 'offset.' .. opt.info .. '.mat.torch', pos)
end


main()

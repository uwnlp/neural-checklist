require 'cutorch'

local RecipeDataMinibatchLoader = require 'RecipeDataMinibatchLoader'
local neural_checklist_model = require 'neural_checklist_model'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a neural checklist model')
cmd:text()
cmd:text('Options')
cmd:option('-rnn_size',200,'Size of text generating RNN hidden state')
cmd:option('-batch_size',20, 'number of recipes to train on in parallel')
cmd:option('-dropout', 0.0, 'Amount of dropout')
cmd:option('-num_layers', 1, 'Number of layers for the title encoder and language model')
cmd:option('-init_weight', 0.35, 'Initial weight range for parameters [-x, x]')
cmd:option('-rnn_type', 'gru', 'Type of language model (GRU, LSTM, RNN)')
cmd:option('-evidence_type', 0, '0 = all evidence, 1 = no evidence')
cmd:option('-lm_only', false, '')
cmd:option('-sumnotmean', true, '')
cmd:option('-switchmul', 5.0, '')
cmd:option('-switch_temperature', 2.0, '')
cmd:option('-attention_temperature', 2.0, '')
cmd:option('-end_loss_mul', 100, '')
cmd:option('-dec_rate', false, '')
cmd:option('-info', 'newtmp', '')
cmd:option('-epochs', 35, "Number of epochs to train.")
cmd:option('-embeddings', '', '')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-train_data_dir', '', 'Training data directory')
cmd:option('-dev_data_dir', '', 'Dev set data directory')
cmd:option('-data_file_info', 'v1.', 'Files version info tag')
cmd:option('-checkpoint_dir', '', 'output directory where checkpoints get written')
cmd:option('-gpuidx', 1, 'Index of GPU on which job should be executed.')
cmd:option('-learningRate', 0.5, '')
cmd:option('-max_lm_grad_norm', 5, '')
cmd:option('-max_item_grad_norm', 5, '')
cmd:option('-max_goal_grad_norm', 5, '')
cmd:option('-sentences_to_train', 0, 'Number of sentences to train at a time, 0 for all.')
cmd:option('-model', '', '')
cmd:option('-startepoch', 1, '')

cmd:text()

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
cutorch.setDevice(opt.gpuidx)

scorefile = io.open('score_neural_checklist_models.' .. opt.rnn_type .. '.info' .. opt.info .. '.dev', 'w')

function run_test(model, data_loader, epoch, last_valid_loss)
   local loss = 0
   local tag_loss = 0
   for i=1,data_loader.nvalid do
      sys.tic()
         local text, goal, agenda, batch_len, ref_types, true_new_item_atten, true_used_item_atten = data_loader:next_batch(2)
         local state = {text = text,
                        goal = goal,
                        agenda = agenda,
                        batch_len = batch_len,
                        ref_types = ref_types,
                        true_new_item_atten = true_new_item_atten,
                        true_used_item_atten = true_used_item_atten}


         local avg_err, end_err = model:dev_forward_pass(state)
         loss = loss + avg_err
         tag_loss = tag_loss + end_err
         print(string.format(
                  "[Loss: %f Epoch: %d Position: %d Rate: %f Time: %f, Step Len: %d]",
                  avg_err,
                  epoch,
                  i,
                  opt.learningRate,
                  sys.toc(),
                  state.batch_len[1]
         ))
         sys.tic()
   end
   scorefile:write(loss .. '\n')
   print(string.format("[VALID EPOCH : %d LOSS: %f TOTAL: %d]",
                      epoch, loss / data_loader.nvalid, data_loader.nvalid)) 
   return loss, tag_loss
end

local function train_model(model, data_loader, max_info)
   print('train model')
   local epoch = opt.startepoch
   local step = 0
   local start_time = sys.tic()
   local last_valid_loss = 1e9
   local last_valid_tag_loss = 1e9
--   if opt.startepoch ~= 1 then
--      run_test(model, data_loader, 1, last_valid_loss)
--      run_test(model, data_loader, (epoch - 1), last_valid_loss)
--      os.exit()
--   end
   print(data_loader.nvalid)
      local total = 0
   for epoch = opt.startepoch,opt.epochs do
      print('epoch: ' .. epoch)
      local curr = 1
--   for step_idx = 1, max_info.num_steps 
      local loss = 0
      local epoch_loss = 0
      local total = 0
      for i=1,data_loader.ntrain do
         step = step + 1
         if i % 10 == 0 then
            print('   step: ' .. i .. '/' .. data_loader.ntrain)
         end
         if i % 1000 == 0 then
            collectgarbage()
         end
         local text, goal, agenda, batch_len, ref_types, true_new_item_atten, true_used_item_atten = data_loader:next_batch(1)
         local state = {text = text,
                        goal = goal,
                        agenda = agenda,
                        batch_len = batch_len,
                        ref_types = ref_types,
                        true_new_item_atten = true_new_item_atten,
                        true_used_item_atten = true_used_item_atten}

         local avg_err = model:forward_pass(state)
         loss = loss + avg_err
         epoch_loss = epoch_loss + avg_err
         model:backward_pass(state, opt.batch_size)
         print(string.format(
                  "[Loss: %f Epoch: %d Position: %d Rate: %f Time: %f, Step Len: %d]",
                  avg_err * opt.batch_size,
                  epoch,
                  i * opt.batch_size,
                  opt.learningRate,
                  sys.toc(),
                  state.batch_len[1]
         ))
         sys.tic()
         loss = 0
         total = total + opt.batch_size
      --end
   end
   local sum_loss = 0
--      print(string.format("[EPOCH : %d LOSS: %f TOTAL: %d]",
--                          epoch, epoch_loss / total, total)) 
      local new_loss, new_tag_loss = run_test(model, data_loader, epoch, last_valid_loss)
      sum_loss = sum_loss + new_loss
--      if sum_loss > last_valid_loss then
--         opt.learningRate = opt.learningRate / 2
--      end
      if opt.dec_rate then
         if sum_loss > last_valid_loss or (sum_loss == last_valid_loss and new_tag_loss > last_valid_tag_loss) then
            opt.learningRate = opt.learningRate / 2
         end
      end
      last_valid_loss = sum_loss
      last_valid_tag_loss = new_tag_loss
      if opt.info == '' then
         model:save_model(epoch, (opt.rnn_type .. '.' .. math.floor(last_valid_loss / data_loader.nvalid) .. '.' .. math.floor(new_tag_loss / data_loader.nvalid)))
      else
         model:save_model(epoch, (opt.rnn_type .. '.' .. opt.info .. '.' .. math.floor(last_valid_loss / data_loader.nvalid) .. '.' .. math.floor(new_tag_loss / data_loader.nvalid)))
      end
end
end

local function initialize_model(max_info, dict)
   print('Initializing model...')
   if opt.model == '' then
      local model = neural_checklist_model:new(opt, max_info, dict)
      print('Done.')
      return model
   else
      local model = neural_checklist_model:load_from_point(opt, opt.model, opt.startepoch - 1)
      print('Done.')
      return model
   end
end

local function save_model(model, epoch)
   model:save_models(epoch)
end

local function main()
   local data_loader = RecipeDataMinibatchLoader.create(opt.train_data_dir, opt.dev_data_dir, opt.batch_size, false, opt.data_file_info, opt, true)
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

   protos = {}
   model = initialize_model(max_info, data_loader.dict)
   collectgarbage()
   train_model(model, data_loader, max_info)
   scorefile:close()
end

main()


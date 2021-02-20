import time
import datetime
import numpy as np
import copy
import torch
from collections import defaultdict
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from covid_tools import *

################
# Class designed by Zijun Long for classifying Covid-19 related tweets
# Email: 2452593L@student.gla.ac.uk
# Version: 1.00 Update date: 16/09/2020
# Acknowlegement: Some codes used to initialise BERT are from BERT tutorial(https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz_IxX23biCPP)
################

class Bertnn:
  def __init__(self, train_inputs, train_labels, train_masks, validation_inputs, validation_labels, validation_masks,
               test_batch, batch_size=16, epochs=4, lr=None, category_number=2, categories=None,evluate_methods=None ):
    '''This method is used to initilse some varibles to train and validate BERT model'''
    # Identify the training data set, including inputs, labels and attension mask
    self.train_inputs = train_inputs
    self.train_labels = train_labels
    self.train_masks = train_masks
    self.category_number = category_number
    # Choose one correct version of evluation methods
    if evluate_methods == None:
      self.evluate_methods = Evaluate_model_nn()
    else:
      self.evluate_methods = evluate_methods
      # Identify name for all categories
    if categories == None:
      self.categories = ['GoodsServices','InformationWanted','Volunteer','MovePeople','EmergingThreats','NewSubEvent','ServiceAvailable','Advice','Any']
    else:
      self.categories = categories

    # Set up training parameters
    self.batch_size = batch_size  # batch size of training
    self.epochs = epochs  # Total training epochs
    self.lr = lr  # Learning rate

    # Identify GPU to use
    if torch.cuda.is_available():
      self.device = torch.device("cuda")
      print('There are %d GPU(s) available.' % torch.cuda.device_count())
      print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
      print('No GPU available, using the CPU instead.')
      self.device = torch.device("cpu")

    # Set the validation dataloader
    validation_inputs = torch.tensor(validation_inputs).to(torch.int64)  # Tensor these varibles
    validation_labels = torch.tensor(validation_labels).to(torch.int64)
    validation_masks = torch.tensor(validation_masks).to(torch.int64)
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    self.validation_dataloader = self.createDataloader(validation_data)

    # Set the testing dataloader
    test_inputs = torch.tensor(test_batch[0]).to(torch.int64)
    test_labels = torch.tensor(test_batch[1]).to(torch.int64)
    test_masks = torch.tensor(test_batch[2]).to(torch.int64)
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    self.test_dataloader = self.createDataloader(test_data)

    # Define the optimizer

    # Initialise the BERT model
    self.iniModel()

  def terminate(self):
    '''This method is used to terminate BERT model and delete varbiles no longer used'''
    # delete all dataloader after evluation
    del self.validation_dataloader, self.test_dataloader
    torch.cuda.empty_cache()

  def createDataloader(self, data):
    '''This method is used to create Dataloader for training, beacuse the default mode of
    Pytorch is batch mode
    data should contains inputs, labels, attensions mask in a list
    '''
    # Shuffle all samples
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
    return dataloader

  def iniModel(self):
    '''This method is used to initilise the BERT model from API of hungging face'''
    # Initialise the model
    torch.cuda.empty_cache()
    self.model = BertForSequenceClassification.from_pretrained(
      "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
      num_labels=self.category_number,  # The number of output labels--2 for binary classification.
      # You can increase this for multi-class tasks.
      output_attentions=False,  # Whether the model returns attentions weights.
      output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    # send model to GPU
    self.model.cuda()

  def updateParameters(self, train_dataloader, lr=None):
    '''This method is used to update some hyper-paramters of BERT'''
    total_steps = len(train_dataloader) * self.epochs
    # If new leraning rate is passed to this method, then update it
    if lr != None:
      self.optimizer = AdamW(self.model.parameters(),
                             lr=lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                             eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                             )
    else:
      self.optimizer = AdamW(self.model.parameters(),
                             lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                             eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                             )
    # Create the learning rate scheduler. Decay the learning rate step by step
    self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                     num_warmup_steps=0,  # Default value in run_glue.py
                                                     num_training_steps=total_steps)

  def searchUpsample(self, increaseTimes,lr=None):
    '''This method is used to try different over-sample factor to train BERT model'''
    # build dict and list to store metrics
    all_metrics = {}  # all metrics
    all_test_metrics = {}  # all test metrcis
    bestF1 = defaultdict()  # store the best F1 score
    bestF1['bestScore'] = 0
    bestF1['bestIncreaseTime'] = None  # store the best over-sample factor
    for increaseTime in range(0, increaseTimes, 10):

      print('Current increase %d times' % (increaseTime))
      # protect the original dataset
      train_inputs_upsample = list(copy.deepcopy(self.train_inputs))
      train_labels_upsample = list(copy.deepcopy(self.train_labels))
      train_masks_upsample = list(copy.deepcopy(self.train_masks))

      # up-sample the label for each category
      for category in range(self.category_number):
        # Does not up-sample the majority class
        if category == 0:
          continue
        print("Current is processing category %d" % (category))
        # build the up-sample data set
        train_inputs_upsample, train_labels_upsample, train_masks_upsample = Up_sample_Bertnn.upSample(train_inputs_upsample,
                                                                                           train_labels_upsample,
                                                                                           train_masks_upsample,
                                                                                           category, increaseTime)
        print()

      # Repost stastistic of this category before and after over=sampling
      for category in range(self.category_number):
        print('Currrent is processing on categorie %s' % (self.categories[category]))
        print()

        print('Before up-sample, the ratio of current category and all samples')
        print(np.count_nonzero(self.train_labels == category))
        print(np.count_nonzero(self.train_labels == category) / len(self.train_labels))
        print()

        print('After up-sample, the ratio of current category and all samples')
        print(np.count_nonzero(np.array(train_labels_upsample) == category))
        print(np.count_nonzero(np.array(train_labels_upsample) == category) / len(train_labels_upsample))
        print()

      # Use up-sample data set to train nerual network

      train_inputs = torch.tensor(np.array(train_inputs_upsample)).to(torch.int64)
      train_labels = torch.tensor(np.array(train_labels_upsample)).to(torch.int64)
      train_masks = torch.tensor(np.array(train_masks_upsample)).to(torch.int64)

      # Create the DataLoader for our training set.
      train_data = TensorDataset(train_inputs, train_masks, train_labels)
      train_dataloader = self.createDataloader(train_data)

      # Number of training epochs (authors recommend between 2 and 4)
      self.epochs = 4

      # Re-initilise the Bert model
      self.iniModel()
      if self.lr != None:
        self.updateParameters(train_dataloader, lr)
      else:
        self.updateParameters(train_dataloader)
      # Use over-sample dataset to train and validate BERT model
      all_metrics[increaseTime] = self.evluate_methods.evalTrain(train_dataloader, self.validation_dataloader)
      # Store return metrics result
      all_test_metrics[increaseTime] = self.evluate_methods.evalTest()

      # Delete over-sample dataset
      del self.model, train_inputs, train_labels, train_masks, train_data, train_dataloader

    # Delete BERT mdoel
    self.terminate()
    return all_metrics, all_test_metrics

  def upSampleTrain(self, increaseTime):
    '''This method is design to use a over-sample method to train BERT model and evaluate it'''

    print('Current increase %d times to train' % (increaseTime))
    # protect the original dataset
    # protect the original dataset
    train_inputs_upsample = copy.deepcopy(self.train_inputs)
    train_labels_upsample = copy.deepcopy(self.train_labels)
    train_masks_upsample = copy.deepcopy(self.train_masks)
    # up-sample the label for each category
    for category in range(4):
      print("Current is processing category %d" % (category))
      # build the up-sample data set
      train_inputs_upsample, train_labels_upsample, train_masks_upsample = Up_sample_Bertnn.upSample(train_inputs_upsample,
                                                                                         train_labels_upsample,
                                                                                         train_masks_upsample, category,
                                                                                         increaseTime)
      print()



    # Use up-sample data set to train nerual network
    print(np.array(train_inputs_upsample).shape)
    train_inputs = torch.tensor(np.array(train_inputs_upsample)).to(torch.int64)
    train_labels = torch.tensor(np.array(train_labels_upsample)).to(torch.int64)
    train_masks = torch.tensor(np.array(train_masks_upsample)).to(torch.int64)

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_dataloader = self.createDataloader(train_data)

    # Number of training epochs (authors recommend between 2 and 4)
    self.epochs = 2

    # Re-initilise the Bert model
    self.iniModel()
    # if new learning rate is passed to this method, then udpate the learning rate
    if self.lr != None:
      self.updateParameters(train_dataloader, self.lr)
    else:
      self.updateParameters(train_dataloader)

    return self.evluate_methods.evalTrain(train_dataloader, self.validation_dataloader)

class Scoring_nn(Scoring):

  def flat_accuracy(self, preds, labels):
    '''This method is used to calculate accuracy'''
    # Choose the category with highest possibility
    pred_flat = np.array(np.argmax(preds, axis=1).flatten())
    # print('flat')
    # print(pred_flat)
    labels_flat = np.array(labels.flatten())
    # print(labels_flat)
    # print(np.sum(pred_flat == labels_flat))
    return float(np.sum(pred_flat == labels_flat) / len(labels_flat))

  def flat_metrics(self, preds, labels, category):
    '''This method is used to calculate serveral metrics, including precsion, recall
    , F1 score'''
    # Choose the category with highest possibility
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    #  Get indexs where prediction is correct
    idxs = np.where(pred_flat == category)[0]
    # print('idxs length',len(idxs))
    # calculate the number of true labels which belong to wanted category
    true_labels_num = len(np.where(labels_flat == category)[0])

    # calculate the number of correct predictions which belong to wanted category
    correct_num = 0
    for idx in idxs:
      if pred_flat[idx] == labels_flat[idx]:
        correct_num += 1
    # print('correct number ',correct_num)
    if len(idxs) == 0:
      precision = 0
    else:
      precision = correct_num / len(idxs)

    if true_labels_num == 0:
      recall = 0
    else:
      recall = correct_num / true_labels_num

    if precision == 0 and recall == 0:
      F1 = 0
    else:
      F1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, F1

class Up_sample_Bertnn(Up_sample):
  def upSample(self, trainSet, trainL, train_masks, category, increaseTimes):
    '''This method is designed to build the over-sample data set'''
    # 'category' is an int to indicate the label

    # Build up-sample
    trainSet_temp = list(copy.deepcopy(trainSet))
    label_temp = list(copy.deepcopy(trainL))
    masks_temp = list(copy.deepcopy(train_masks))
    # trainSet_temp = trainSet
    # label_temp = trainL
    # masks_temp = train_masks
    # print some stastitcs
    print('Before up-sample, there are %d samples' % (len(label_temp)))
    idxs = np.where(np.array(label_temp) == category)[0]
    print('There are %d sampled in this category.' % (len(idxs)))

    # interate every samples
    for idx in list(idxs):
      # Does not duplicate the majority class
      if category == 0:  # or category == 1 :
        break

      # Duplicate inputs, labels, attension mask (increaseTimes) times
      for j in range(increaseTimes):
        trainSet_temp.append(trainSet[idx])
        label_temp.append(category)
        masks_temp.append(train_masks[idx])

    print('After up-sample, there are %d samples' % (len(label_temp)))

    return trainSet_temp, label_temp, masks_temp

class Evaluate_model_nn(Evaluate_model):
  def evalTrain(self,model, train_dataloader, validation_dataloader, epochs,
                category_number, optimizer, scheduler,):
    ''' This method is used to train and validate BERT model and return some metrics'''
    # Store some metrics.
    train_loss_values = []
    train_acc_all = []
    train_precision_all = []
    train_recall_all = []
    train_f1_all = []
    val_acc_all = []
    val_precision_all = []
    val_recall_all = []
    val_f1_all = []
    avg_train_f1_all, avg_train_recall_all, avg_train_precision_all = [], [], []
    avg_val_f1_all, avg_val_recall_all, avg_val_precision_all = [], [], []
    # Identify GPU to use
    if torch.cuda.is_available():
      device = torch.device("cuda")
      print('There are %d GPU(s) available.' % torch.cuda.device_count())
      print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
      print('No GPU available, using the CPU instead.')
      device = torch.device("cpu")

    # For each epoch...
    for epoch_i in range(0, epochs):

      # ========================================
      #               Training
      # ========================================

      # Perform one full pass over the training set.
      print("")
      print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
      print('Training...')

      # Measure how long the training epoch takes.
      t0 = time.time()
      # Reset the total loss for this epoch.
      total_loss, nb_train_steps, train_accuracy = 0, 0, 0
      train_precision, train_recall, train_f1 = {}, {}, {}
      for i in range(category_number):
        train_precision[i] = 0
        train_recall[i] = 0
        train_f1[i] = 0

      model.train()
      # For each batch of training data...
      for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
          # Calculate elapsed time in minutes.
          elapsed = Scoring_nn.format_time(time.time() - t0)

          # Report progress.
          print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Clear gradient of last step
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        # This call will not return loss value
        logits = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask)[0]
        # The call to `model` always returns a tuple, so we need to pull the
        # loss value out of the tuple.
        loss = outputs[0]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

        # Calculate the accuracy and precision
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_train_accuracy = Scoring_nn.flat_accuracy(self, logits, label_ids)

        # Build the dict for each metrics and store metrics for each category
        tmp_train_precision = {}
        tmp_train_recall = {}
        tmp_train_f1 = {}
        for i in range(category_number):
          tmp_train_precision[i] = 0
          tmp_train_recall[i] = 0
          tmp_train_f1[i] = 0

        # Use flat_metrics to get precision, recall, f1 score
        for k in range(category_number):
          tmp_train_precision[k], tmp_train_recall[k], tmp_train_f1[k] = Scoring_nn.flat_metrics(self, logits, label_ids, k)

        # Accumulate the result.
        train_accuracy += tmp_train_accuracy
        for k in range(category_number):
          train_precision[k] += tmp_train_precision[k]
          train_recall[k] += tmp_train_recall[k]
          train_f1[k] += tmp_train_f1[k]
        nb_train_steps += 1

      # Calculate the average loss over the training data.
      avg_train_loss = total_loss / len(train_dataloader)

      # Store the loss value for plotting the learning curve.
      train_loss_values.append(avg_train_loss)

      print("")
      print("  Average training loss: {0:.2f}".format(avg_train_loss))
      print("  Accuracy: {0:.2f}".format(train_accuracy / nb_train_steps))

      # Calculate the average value for each metrics
      precision_avg, recall_avg, f1_avg = 0, 0, 0
      for k in range(category_number):
        print("Category: %d" % (k))
        train_precision[k] = train_precision[k] / nb_train_steps
        precision_avg += train_precision[k]
        train_recall[k] = train_recall[k] / nb_train_steps
        recall_avg += train_recall[k]
        train_f1[k] = train_f1[k] / nb_train_steps
        f1_avg += train_f1[k]
        print("  Precision: {0:.2f}".format(train_precision[k]))
        print("  Recall: {0:.2f}".format(train_recall[k]))
        print("  F1: {0:.2f}".format(train_f1[k]))

      # Output metircs
      print("The average precision is: {0:.4f}".format(precision_avg / category_number))
      print("The average recall is: {0:.4f}".format(recall_avg / category_number))
      print("The average f1 is: {0:.4f}".format(f1_avg / category_number))
      print("  Training epcoh took: {:}".format(Scoring_nn.format_time(self, time.time() - t0)))

      # Store them in a list
      train_precision_all.append(train_precision)
      train_loss_values.append(avg_train_loss)
      train_acc_all.append(train_accuracy / nb_train_steps)
      train_recall_all.append(train_recall)
      train_f1_all.append(train_f1)
      avg_train_f1_all.append(f1_avg / 4)
      avg_train_recall_all.append(recall_avg / 4)
      avg_train_precision_all.append(precision_avg / 4)

      # ========================================
      #               Validation
      # ========================================

      print("")
      print("Running Validation...")

      t0 = time.time()
      # Put the model in evaluation mode--the dropout layers behave differently
      # during evaluation.
      model.eval()

      # Tracking variables
      eval_accuracy = 0
      nb_eval_steps, nb_eval_examples = 0, 0
      eval_precision, eval_recall, eval_f1 = {}, {}, {}
      for i in range(category_number):
        eval_precision[i] = 0
        eval_recall[i] = 0
        eval_f1[i] = 0

      # Evaluate data for one epoch
      for batch in validation_dataloader:

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():

          # Forward pass, calculate logit predictions.
          # This will return the logits rather than the loss because we have
          # not provided labels.
          outputs = model(b_input_ids,
                          token_type_ids=None,
                          attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = Scoring_nn.flat_accuracy(self, logits, label_ids)

        # Build the dict for each metrics and store metrics for each category
        tmp_eval_precision = {}
        tmp_eval_recall = {}
        tmp_eval_f1 = {}
        for i in range(category_number):
          tmp_eval_precision[i] = 0
          tmp_eval_recall[i] = 0
          tmp_eval_f1[i] = 0

        for k in range(category_number):
          tmp_eval_precision[k], tmp_eval_recall[k], tmp_eval_f1[k] = Scoring_nn.flat_metrics(self, logits, label_ids, k)

        # Accumulate metircs.
        eval_accuracy += tmp_eval_accuracy

        for k in range(category_number):
          eval_precision[k] += tmp_eval_precision[k]
          eval_recall[k] += tmp_eval_recall[k]
          eval_f1[k] += tmp_eval_f1[k]

        # Track the number of batches
        nb_eval_steps += 1

      # Report the final accuracy for this validation run.
      # print(eval_accuracy)
      print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))

      # Calculate the average value for each metrics
      precision_avg, recall_avg, f1_avg = 0, 0, 0
      for k in range(category_number):
        print("Category: %d" % (k))
        eval_precision[k] = eval_precision[k] / nb_eval_steps
        precision_avg += eval_precision[k]
        eval_recall[k] = eval_recall[k] / nb_eval_steps
        recall_avg += eval_recall[k]
        eval_f1[k] = eval_f1[k] / nb_eval_steps
        f1_avg += eval_f1[k]
        print("  Precision: {0:.4f}".format(eval_precision[k]))
        print("  Recall: {0:.4f}".format(eval_recall[k]))
        print("  F1: {0:.4f}".format(eval_f1[k]))

      # Display these metrics
      print("The average precision is: {0:.4f}".format(precision_avg / category_number))
      print("The average recall is: {0:.4f}".format(recall_avg / category_number))
      print("The average f1 is: {0:.4f}".format(f1_avg / category_number))
      print("  Validation took: {:}".format(Scoring_nn.format_time(self, time.time() - t0)))

      # Append metrics to list and store them
      val_precision_all.append(eval_precision)
      val_recall_all.append(eval_recall)
      val_f1_all.append(eval_f1)
      val_acc_all.append(eval_accuracy / nb_eval_steps)
      avg_val_f1_all.append(f1_avg / category_number)
      avg_val_recall_all.append(recall_avg / category_number)
      avg_val_precision_all.append(recall_avg / category_number)

    print("")
    print("Training complete!")

    # Store all metrics duing training and validation. Return it
    metrics_all = defaultdict()
    metrics_all['train_loss_values'], metrics_all['train_acc_all'], metrics_all[
      'train_precision_all'] = train_loss_values, train_acc_all, train_precision_all
    metrics_all['train_recall_all'], metrics_all['train_f1_all'], metrics_all['val_acc_all'], metrics_all[
      'val_precision_all'], metrics_all['val_recall_all'], metrics_all[
      'val_f1_all'] = train_recall_all, train_f1_all, val_acc_all, val_precision_all, val_recall_all, val_f1_all
    metrics_all['avg_train_f1_all'], metrics_all['avg_train_recall_all'], metrics_all[
      ' avg_train_precision_all'] = avg_train_f1_all, avg_train_recall_all, avg_train_precision_all
    metrics_all['avg_val_f1_all'], metrics_all['avg_val_recall_all'], metrics_all[
      'avg_val_precision_all'] = avg_val_f1_all, avg_val_recall_all, avg_val_precision_all
    return metrics_all

  def evalTest(self, model, test_dataloader, category_number):
    '''This method is used to only validate trained BERT model on a test set'''
    # ========================================
    #               Testing
    # ========================================
    print()
    print('Testing...')
    # record the start time
    t0 = time.time()

    # Identify GPU to use
    if torch.cuda.is_available():
      device = torch.device("cuda")
      print('There are %d GPU(s) available.' % torch.cuda.device_count())
      print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
      print('No GPU available, using the CPU instead.')
      device = torch.device("cpu")
    # put model in evaluation model
    model.eval()

    # Build list to store metrics of each epoches
    test_acc_all = []
    test_precision_all = []
    test_recall_all = []
    test_f1_all = []
    test_accuracy = 0
    nb_test_steps, nb_test_examples = 0, 0

    # Build the dict for each metrics and store metrics for each category
    test_precision, test_recall, test_f1 = {}, {}, {}
    for i in range(category_number):
      test_f1[i], test_precision[i], test_recall[i] = 0, 0, 0

    # For each batch of training data...
    for batch in test_dataloader:
      # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)

      # Unpack the inputs from our dataloader
      b_input_ids, b_input_mask, b_labels = batch

      # Telling the model not to compute or store gradients, saving memory and
      # speeding up validation
      with torch.no_grad():

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
      # Get the "logits" output by the model. The "logits" are the output
      # values prior to applying an activation function like the softmax.
      logits = outputs[0]

      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()

      # Calculate the accuracy for this batch of test sentences.
      tmp_test_accuracy = Scoring_nn.flat_accuracy(self, logits, label_ids)

      tmp_test_precision = {}
      tmp_test_recall = {}
      tmp_test_f1 = {}
      for i in range(category_number):
        tmp_test_precision[i] = 0
        tmp_test_recall[i] = 0
        tmp_test_f1[i] = 0

      for k in range(category_number):
        tmp_test_precision[k], tmp_test_recall[k], tmp_test_f1[k] = Scoring_nn.flat_metrics(self, logits, label_ids, k)

      # Accumulate the total accuracy.
      test_accuracy += tmp_test_accuracy

      for k in range(category_number):
        test_precision[k] += tmp_test_precision[k]
        test_recall[k] += tmp_test_recall[k]
        test_f1[k] += tmp_test_f1[k]

      # Track the number of batches
      nb_test_steps += 1

      # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(test_accuracy / nb_test_steps))

    # Calculate the average value for each metrics
    precision_avg, recall_avg, f1_avg = 0, 0, 0
    for k in range(category_number):
      print("Category: %d" % (k))
      test_precision[k] = test_precision[k] / nb_test_steps
      precision_avg += test_precision[k]
      test_recall[k] = test_recall[k] / nb_test_steps
      recall_avg += test_recall[k]
      test_f1[k] = test_f1[k] / nb_test_steps
      f1_avg += test_f1[k]
      print("  Precision: {0:.4f}".format(test_precision[k]))
      print("  Recall: {0:.4f}".format(test_recall[k]))
      print("  F1: {0:.4f}".format(test_f1[k]))
    # Report other metrics
    print("The average precision is: {0:.4f}".format(precision_avg / category_number))
    print("The average recall is: {0:.4f}".format(recall_avg / category_number))
    print("The average f1 is: {0:.4f}".format(f1_avg / category_number))
    print("  Testing took: {:}".format(Scoring_nn.format_time(self, time.time() - t0)))

    # Store them in a list
    test_precision_all.append(test_precision)
    test_recall_all.append(test_recall)
    test_f1_all.append(test_f1)
    test_acc_all.append(test_accuracy / nb_test_steps)

    return {'precision': test_precision_all, 'recall': test_recall_all, 'f1': test_f1_all, 'acc': test_acc_all}

class Evaluate_model_nn_rectified_binary(Evaluate_model_nn):
  def evalTrain(self, model, train_dataloader, validation_dataloader, epochs,
                category_number, optimizer, scheduler,
                optimizer_small, scheduler_small, best_F1_avg):
    ''' This method is used to train and validate BERT model and return some metrics'''
    # Store some metrics.
    train_loss_values = []
    train_acc_all = []
    train_precision_all = []
    train_recall_all = []
    train_f1_all = []
    val_acc_all = []
    val_precision_all = []
    val_recall_all = []
    val_f1_all = []
    avg_train_f1_all, avg_train_recall_all, avg_train_precision_all = [], [], []
    avg_val_f1_all, avg_val_recall_all, avg_val_precision_all = [], [], []
    current_best_F1_avg = best_F1_avg

    # Identify GPU to use
    if torch.cuda.is_available():
      device = torch.device("cuda")
      print('There are %d GPU(s) available.' % torch.cuda.device_count())
      print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
      print('No GPU available, using the CPU instead.')
      device = torch.device("cpu")

    # For each epoch...
    for epoch_i in range(0, epochs):

      # ========================================
      #               Training
      # ========================================

      # Perform one full pass over the training set.
      print("")
      print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
      print('Training...')

      # Measure how long the training epoch takes.
      t0 = time.time()
      # Reset the total loss for this epoch.
      total_loss, nb_train_steps, train_accuracy = 0, 0, 0
      train_precision, train_recall, train_f1 = {}, {}, {}
      for i in range(category_number):
        train_precision[i] = 0
        train_recall[i] = 0
        train_f1[i] = 0

      model.train()
      # For each batch of training data...
      for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
          # Calculate elapsed time in minutes.
          elapsed = Scoring_nn.format_time(self,  time.time() - t0)

          # Report progress.
          print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Clear gradient of last step
        model.zero_grad()
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        # This call will not return loss value
        logits = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask)[0]

        # The call to `model` always returns a tuple, so we need to pull the
        # loss value out of the tuple.
        loss = outputs[0]

        # total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

        one_batch_inputs_ids = batch[0].detach().cpu().numpy()
        one_batch_masks = batch[1].detach().cpu().numpy()
        one_batch_labels = batch[2].detach().cpu().numpy()

        # incremental rectified training for each category separately

        for category in range(category_number):

          # Only training for minority class
          if category == 0:
            continue

          # identify indexs for positive samples
          inputs_ids_one_category = []
          labels_one_category = []
          mask_one_category = []
          # if there is no positive samples in the batch, go to next category
          length_one_category = len(np.where(one_batch_labels == category)[0])
          if length_one_category == 0:
            continue

          # collect input, labels, attension mask for positive samples
          for idx in np.where(one_batch_labels == category)[0]:
            inputs_ids_one_category.append(one_batch_inputs_ids[idx])
            labels_one_category.append(one_batch_labels[idx])
            mask_one_category.append(one_batch_masks[idx])

          # Tensor all data
          inputs_ids_one_category = torch.tensor(np.array(inputs_ids_one_category)).to(torch.int64).to(device)
          mask_one_category = torch.tensor(np.array(mask_one_category)).to(torch.int64).to(device)
          labels_one_category = torch.tensor(np.array(labels_one_category)).to(torch.int64).to(device)

          # clear gradients
          model.zero_grad()
          # Calculate the loss
          outputs_one_category = model(inputs_ids_one_category,
                                       token_type_ids=None,
                                       attention_mask=mask_one_category,
                                       labels=labels_one_category,
                                       output_hidden_states=True)
          loss_one_category = outputs_one_category[0]

          # Perform a backward pass to calculate the gradients.
          loss_one_category.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          total_loss += loss_one_category.item()

          # Choose corespoding optimizer of each class to optimize hyper-parameters
          if category == 0:
            # print('Invoke 0')
            optimizer_small.step()
            scheduler_small.step()
          elif category == 1:
            # print('Invoke 1')
            optimizer_small.step()
            scheduler_small.step()

          # delete tempory valbires
          del inputs_ids_one_category, mask_one_category, labels_one_category, outputs_one_category, loss_one_category
          torch.cuda.empty_cache()
          # Calculate the accuracy and precision
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_train_accuracy = Scoring_nn.flat_accuracy(self, logits, label_ids)

        # Build the dict for each metrics and store metrics for each category
        tmp_train_precision = {}
        tmp_train_recall = {}
        tmp_train_f1 = {}
        for i in range(category_number):
          tmp_train_precision[i] = 0
          tmp_train_recall[i] = 0
          tmp_train_f1[i] = 0

        # Use flat_metrics to get precision, recall, f1 score
        for k in range(category_number):
          tmp_train_precision[k], tmp_train_recall[k], tmp_train_f1[k] = Scoring_nn.flat_metrics(self, logits, label_ids, k)

        # Accumulate the result.
        train_accuracy += tmp_train_accuracy
        for k in range(category_number):
          train_precision[k] += tmp_train_precision[k]
          train_recall[k] += tmp_train_recall[k]
          train_f1[k] += tmp_train_f1[k]
        nb_train_steps += 1
      # Calculate the average loss over the training data.
      avg_train_loss = total_loss / len(train_dataloader)

      # Store the loss value for plotting the learning curve.
      train_loss_values.append(avg_train_loss)

      print("")
      print("  Average training loss: {0:.2f}".format(avg_train_loss))
      print("  Accuracy: {0:.2f}".format(train_accuracy / nb_train_steps))

      # Calculate the average value for each metrics
      precision_avg, recall_avg, f1_avg = 0, 0, 0
      for k in range(category_number):
        print("Category: %d" % (k))
        train_precision[k] = train_precision[k] / nb_train_steps
        precision_avg += train_precision[k]
        train_recall[k] = train_recall[k] / nb_train_steps
        recall_avg += train_recall[k]
        train_f1[k] = train_f1[k] / nb_train_steps
        f1_avg += train_f1[k]
        print("  Precision: {0:.2f}".format(train_precision[k]))
        print("  Recall: {0:.2f}".format(train_recall[k]))
        print("  F1: {0:.2f}".format(train_f1[k]))

      # Output metircs
      print("The average precision is: {0:.4f}".format(precision_avg / category_number))
      print("The average recall is: {0:.4f}".format(recall_avg / category_number))
      print("The average f1 is: {0:.4f}".format(f1_avg / category_number))
      print("  Training epcoh took: {:}".format(Scoring_nn.format_time(self, time.time() - t0)))

      # Store them in a list
      train_precision_all.append(train_precision)
      train_loss_values.append(avg_train_loss)
      train_acc_all.append(train_accuracy / nb_train_steps)
      train_recall_all.append(train_recall)
      train_f1_all.append(train_f1)
      avg_train_f1_all.append(f1_avg / 4)
      avg_train_recall_all.append(recall_avg / 4)
      avg_train_precision_all.append(precision_avg / 4)

      # delete varibles no longer use
      del b_input_ids, b_input_mask, b_labels, outputs, logits
      torch.cuda.empty_cache()
      # ========================================
      #               Validation
      # ========================================

      print("")
      print("Running Validation...")

      t0 = time.time()

      # Put the model in evaluation mode--the dropout layers behave differently
      # during evaluation.
      model.eval()

      # Tracking variables
      eval_accuracy = 0
      nb_eval_steps, nb_eval_examples = 0, 0
      eval_precision, eval_recall, eval_f1 = {}, {}, {}
      for i in range(category_number):
        eval_precision[i] = 0
        eval_recall[i] = 0
        eval_f1[i] = 0

      # Evaluate data for one epoch
      for batch in validation_dataloader:

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():

          # Forward pass, calculate logit predictions.
          # This will return the logits rather than the loss because we have
          # not provided labels.
          outputs = model(b_input_ids,
                          token_type_ids=None,
                          attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = Scoring_nn.flat_accuracy(self, logits, label_ids)

        # Build the dict for each metrics and store metrics for each category
        tmp_eval_precision = {}
        tmp_eval_recall = {}
        tmp_eval_f1 = {}
        for i in range(category_number):
          tmp_eval_precision[i] = 0
          tmp_eval_recall[i] = 0
          tmp_eval_f1[i] = 0

        for k in range(category_number):
          tmp_eval_precision[k], tmp_eval_recall[k], tmp_eval_f1[k] = Scoring_nn.flat_metrics(self, logits, label_ids, k)

        # Accumulate metircs.
        eval_accuracy += tmp_eval_accuracy

        for k in range(category_number):
          eval_precision[k] += tmp_eval_precision[k]
          eval_recall[k] += tmp_eval_recall[k]
          eval_f1[k] += tmp_eval_f1[k]

        # Track the number of batches
        nb_eval_steps += 1

        del b_input_ids, b_input_mask, b_labels, outputs, logits, label_ids
        torch.cuda.empty_cache()
        # Report the final accuracy for this validation run.
      # print(eval_accuracy)
      print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))

      # Calculate the average value for each metrics
      precision_avg, recall_avg, f1_avg = 0, 0, 0
      for k in range(category_number):
        print("Category: %d" % (k))
        eval_precision[k] = eval_precision[k] / nb_eval_steps
        precision_avg += eval_precision[k]
        eval_recall[k] = eval_recall[k] / nb_eval_steps
        recall_avg += eval_recall[k]
        eval_f1[k] = eval_f1[k] / nb_eval_steps
        f1_avg += eval_f1[k]
        print("  Precision: {0:.4f}".format(eval_precision[k]))
        print("  Recall: {0:.4f}".format(eval_recall[k]))
        print("  F1: {0:.4f}".format(eval_f1[k]))

      # Display these metrics
      print("The average precision is: {0:.4f}".format(precision_avg / category_number))
      print("The average recall is: {0:.4f}".format(recall_avg / category_number))
      print("The average f1 is: {0:.4f}".format(f1_avg / category_number))
      print("  Validation took: {:}".format(Scoring_nn.format_time(self, time.time() - t0)))

      # Append metrics to list and store them
      val_precision_all.append(eval_precision)
      val_recall_all.append(eval_recall)
      val_f1_all.append(eval_f1)
      val_acc_all.append(eval_accuracy / nb_eval_steps)
      avg_val_f1_all.append(f1_avg / category_number)
      avg_val_recall_all.append(recall_avg / category_number)
      avg_val_precision_all.append(recall_avg / category_number)

      # Compare the best average f1 score and store the best model
      avg_val_f1 = f1_avg / category_number
      if avg_val_f1 > current_best_F1_avg:
        print("Found new best average F1 score model when validation, old f1 is %f , new f1 is %f" %(best_F1_avg, avg_val_f1))
        torch.save(model.state_dict(),'Best_val_Bert_rectified_binary.pth')
        current_best_F1_avg = avg_val_f1

    print("")
    print("Training complete!")

    # Store all metrics duing training and validation. Return it
    metrics_all = defaultdict()
    metrics_all['train_loss_values'], metrics_all['train_acc_all'], metrics_all[
      'train_precision_all'] = train_loss_values, train_acc_all, train_precision_all
    metrics_all['train_recall_all'], metrics_all['train_f1_all'], metrics_all['val_acc_all'], metrics_all[
      'val_precision_all'], metrics_all['val_recall_all'], metrics_all[
      'val_f1_all'] = train_recall_all, train_f1_all, val_acc_all, val_precision_all, val_recall_all, val_f1_all
    metrics_all['avg_train_f1_all'], metrics_all['avg_train_recall_all'], metrics_all[
      ' avg_train_precision_all'] = avg_train_f1_all, avg_train_recall_all, avg_train_precision_all
    metrics_all['avg_val_f1_all'], metrics_all['avg_val_recall_all'], metrics_all[
      'avg_val_precision_all'] = avg_val_f1_all, avg_val_recall_all, avg_val_precision_all
    return metrics_all, current_best_F1_avg

  def evalTest(self, model, test_dataloader, category_number):
    '''This method is used to only validate trained BERT model on a test set'''
    # ========================================
    #               Testing
    # ========================================
    print()
    print('Testing...')
    # record the start time
    t0 = time.time()

    # Identify GPU to use
    if torch.cuda.is_available():
      device = torch.device("cuda")
      print('There are %d GPU(s) available.' % torch.cuda.device_count())
      print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
      print('No GPU available, using the CPU instead.')
      device = torch.device("cpu")
    # put model in evaluation model
    model.eval()

    # Build list to store metrics of each epoches
    test_acc_all = []
    test_precision_all = []
    test_recall_all = []
    test_f1_all = []
    test_accuracy = 0
    nb_test_steps, nb_test_examples = 0, 0

    # Build the dict for each metrics and store metrics for each category
    test_precision, test_recall, test_f1 = {}, {}, {}
    for i in range(category_number):
      test_f1[i], test_precision[i], test_recall[i] = 0, 0, 0

    # For each batch of training data...
    for batch in test_dataloader:
      # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)

      # Unpack the inputs from our dataloader
      b_input_ids, b_input_mask, b_labels = batch

      # Telling the model not to compute or store gradients, saving memory and
      # speeding up validation
      with torch.no_grad():

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
      # Get the "logits" output by the model. The "logits" are the output
      # values prior to applying an activation function like the softmax.
      logits = outputs[0]

      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()

      # Calculate the accuracy for this batch of test sentences.
      tmp_test_accuracy = Scoring_nn.flat_accuracy(self, logits, label_ids)

      tmp_test_precision = {}
      tmp_test_recall = {}
      tmp_test_f1 = {}
      for i in range(category_number):
        tmp_test_precision[i] = 0
        tmp_test_recall[i] = 0
        tmp_test_f1[i] = 0

      for k in range(category_number):
        tmp_test_precision[k], tmp_test_recall[k], tmp_test_f1[k] = Scoring_nn.flat_metrics(self, logits, label_ids, k)

      # Accumulate the total accuracy.
      test_accuracy += tmp_test_accuracy

      for k in range(category_number):
        test_precision[k] += tmp_test_precision[k]
        test_recall[k] += tmp_test_recall[k]
        test_f1[k] += tmp_test_f1[k]

      # Track the number of batches
      nb_test_steps += 1

      # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(test_accuracy / nb_test_steps))

    # Calculate the average value for each metrics
    precision_avg, recall_avg, f1_avg = 0, 0, 0
    for k in range(category_number):
      print("Category: %d" % (k))
      test_precision[k] = test_precision[k] / nb_test_steps
      precision_avg += test_precision[k]
      test_recall[k] = test_recall[k] / nb_test_steps
      recall_avg += test_recall[k]
      test_f1[k] = test_f1[k] / nb_test_steps
      f1_avg += test_f1[k]
      print("  Precision: {0:.4f}".format(test_precision[k]))
      print("  Recall: {0:.4f}".format(test_recall[k]))
      print("  F1: {0:.4f}".format(test_f1[k]))
    # Report other metrics
    print("The average precision is: {0:.4f}".format(precision_avg / category_number))
    print("The average recall is: {0:.4f}".format(recall_avg / category_number))
    print("The average f1 is: {0:.4f}".format(f1_avg / category_number))
    print("  Testing took: {:}".format(Scoring_nn.format_time(self, time.time() - t0)))

    # Store them in a list
    test_precision_all.append(test_precision)
    test_recall_all.append(test_recall)
    test_f1_all.append(test_f1)
    test_acc_all.append(test_accuracy / nb_test_steps)

    return {'precision': test_precision_all, 'recall': test_recall_all, 'f1': test_f1_all, 'acc': test_acc_all}

class Evaluate_model_nn_rectified_multi_classes(Evaluate_model_nn):
  def evalTrain(self, model, train_dataloader, validation_dataloader, epochs,
                category_number, optimizer, scheduler,
                optimizer_small, optimizer_medium, scheduler_small, scheduler_medium,  best_F1_avg):
    ''' This method is used to train and validate BERT model and return some metrics'''
    # Store some metrics.
    train_loss_values = []
    train_acc_all = []
    train_precision_all = []
    train_recall_all = []
    train_f1_all = []
    val_acc_all = []
    val_precision_all = []
    val_recall_all = []
    val_f1_all = []
    avg_train_f1_all, avg_train_recall_all, avg_train_precision_all = [], [], []
    avg_val_f1_all, avg_val_recall_all, avg_val_precision_all = [], [], []
    current_best_F1_avg = best_F1_avg

    # Identify GPU to use
    if torch.cuda.is_available():
      device = torch.device("cuda")
      print('There are %d GPU(s) available.' % torch.cuda.device_count())
      print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
      print('No GPU available, using the CPU instead.')
      device = torch.device("cpu")

    # For each epoch...
    for epoch_i in range(0, epochs):

      # ========================================
      #               Training
      # ========================================

      # Perform one full pass over the training set.
      print("")
      print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
      print('Training...')

      # Measure how long the training epoch takes.
      t0 = time.time()
      # Reset the total loss for this epoch.
      total_loss, nb_train_steps, train_accuracy = 0, 0, 0
      train_precision, train_recall, train_f1 = {}, {}, {}
      for i in range(category_number):
        train_precision[i] = 0
        train_recall[i] = 0
        train_f1[i] = 0

      model.train()
      # For each batch of training data...
      for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
          # Calculate elapsed time in minutes.
          elapsed = Scoring_nn.format_time(self, time.time() - t0)

          # Report progress.
          print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Clear gradient of last step
        model.zero_grad()
        outputs = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)

        # This call will not return loss value
        logits = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)[0]

        # The call to `model` always returns a tuple, so we need to pull the
        # loss value out of the tuple.
        loss = outputs[0]

        # total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

        one_batch_inputs_ids = batch[0].detach().cpu().numpy()
        one_batch_masks = batch[1].detach().cpu().numpy()
        one_batch_labels = batch[2].detach().cpu().numpy()

        # incremental rectified training for each category separately

        for category in range(category_number):

          # Only training for minority class
          if category == 0:
            continue

          # identify indexs for positive samples
          inputs_ids_one_category = []
          labels_one_category = []
          mask_one_category = []
          # if there is no positive samples in the batch, go to next category
          length_one_category = len(np.where(one_batch_labels == category)[0])
          if length_one_category == 0:
            continue

          # collect input, labels, attension mask for positive samples
          for idx in np.where(one_batch_labels == category)[0]:
            inputs_ids_one_category.append(one_batch_inputs_ids[idx])
            labels_one_category.append(one_batch_labels[idx])
            mask_one_category.append(one_batch_masks[idx])

          # Tensor all data
          inputs_ids_one_category = torch.tensor(np.array(inputs_ids_one_category)).to(torch.int64).to(device)
          mask_one_category = torch.tensor(np.array(mask_one_category)).to(torch.int64).to(device)
          labels_one_category = torch.tensor(np.array(labels_one_category)).to(torch.int64).to(device)

          # clear gradients
          model.zero_grad()
          # Calculate the loss
          outputs_one_category = model(inputs_ids_one_category,
                                            token_type_ids=None,
                                            attention_mask=mask_one_category,
                                            labels=labels_one_category,
                                            output_hidden_states=True)
          loss_one_category = outputs_one_category[0]

          # Perform a backward pass to calculate the gradients.
          loss_one_category.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          total_loss += loss_one_category.item()

          # Choose corespoding optimizer of each class to optimize hyper-parameters
          if category == 0:
            # print('Invoke 0')
            optimizer_small.step()
            scheduler_small.step()
          elif category == 1:
            # print('Invoke 1')
            optimizer_small.step()
            scheduler_small.step()
          elif category == 2:
            # print('Invoke 2')
            optimizer_medium.step()
            scheduler_medium.step()
          else:
            # print('Invoke 3')
            # optimizer_large.step()
            # scheduler_large.step()
            optimizer_small.step()
            scheduler_small.step()

          # delete tempory valbires
          del inputs_ids_one_category, mask_one_category, labels_one_category, outputs_one_category, loss_one_category
          torch.cuda.empty_cache()
          # Calculate the accuracy and precision
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_train_accuracy = Scoring_nn.flat_accuracy(self, logits, label_ids)

        # Build the dict for each metrics and store metrics for each category
        tmp_train_precision = {}
        tmp_train_recall = {}
        tmp_train_f1 = {}
        for i in range(category_number):
          tmp_train_precision[i] = 0
          tmp_train_recall[i] = 0
          tmp_train_f1[i] = 0

        # Use flat_metrics to get precision, recall, f1 score
        for k in range(category_number):
          tmp_train_precision[k], tmp_train_recall[k], tmp_train_f1[k] = Scoring_nn.flat_metrics(self, logits, label_ids, k)

        # Accumulate the result.
        train_accuracy += tmp_train_accuracy
        for k in range(category_number):
          train_precision[k] += tmp_train_precision[k]
          train_recall[k] += tmp_train_recall[k]
          train_f1[k] += tmp_train_f1[k]
        nb_train_steps += 1
      # Calculate the average loss over the training data.
      avg_train_loss = total_loss / len(train_dataloader)

      # Store the loss value for plotting the learning curve.
      train_loss_values.append(avg_train_loss)

      print("")
      print("  Average training loss: {0:.2f}".format(avg_train_loss))
      print("  Accuracy: {0:.2f}".format(train_accuracy / nb_train_steps))

      # Calculate the average value for each metrics
      precision_avg, recall_avg, f1_avg = 0, 0, 0
      for k in range(category_number):
        print("Category: %d" % (k))
        train_precision[k] = train_precision[k] / nb_train_steps
        precision_avg += train_precision[k]
        train_recall[k] = train_recall[k] / nb_train_steps
        recall_avg += train_recall[k]
        train_f1[k] = train_f1[k] / nb_train_steps
        f1_avg += train_f1[k]
        print("  Precision: {0:.2f}".format(train_precision[k]))
        print("  Recall: {0:.2f}".format(train_recall[k]))
        print("  F1: {0:.2f}".format(train_f1[k]))

      # Output metircs
      print("The average precision is: {0:.4f}".format(precision_avg / category_number))
      print("The average recall is: {0:.4f}".format(recall_avg / category_number))
      print("The average f1 is: {0:.4f}".format(f1_avg / category_number))
      print("  Training epcoh took: {:}".format(Scoring_nn.format_time(self, time.time() - t0)))

      # Store them in a list
      train_precision_all.append(train_precision)
      train_loss_values.append(avg_train_loss)
      train_acc_all.append(train_accuracy / nb_train_steps)
      train_recall_all.append(train_recall)
      train_f1_all.append(train_f1)
      avg_train_f1_all.append(f1_avg / 4)
      avg_train_recall_all.append(recall_avg / 4)
      avg_train_precision_all.append(precision_avg / 4)

      # delete varibles no longer use
      del b_input_ids, b_input_mask, b_labels, outputs, logits
      torch.cuda.empty_cache()
      # ========================================
      #               Validation
      # ========================================

      print("")
      print("Running Validation...")

      t0 = time.time()

      # Put the model in evaluation mode--the dropout layers behave differently
      # during evaluation.
      model.eval()

      # Tracking variables
      eval_accuracy = 0
      nb_eval_steps, nb_eval_examples = 0, 0
      eval_precision, eval_recall, eval_f1 = {}, {}, {}
      for i in range(category_number):
        eval_precision[i] = 0
        eval_recall[i] = 0
        eval_f1[i] = 0

      # Evaluate data for one epoch
      for batch in validation_dataloader:

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():

          # Forward pass, calculate logit predictions.
          # This will return the logits rather than the loss because we have
          # not provided labels.
          outputs = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = Scoring_nn.flat_accuracy(self, logits, label_ids)

        # Build the dict for each metrics and store metrics for each category
        tmp_eval_precision = {}
        tmp_eval_recall = {}
        tmp_eval_f1 = {}
        for i in range(category_number):
          tmp_eval_precision[i] = 0
          tmp_eval_recall[i] = 0
          tmp_eval_f1[i] = 0

        for k in range(category_number):
          tmp_eval_precision[k], tmp_eval_recall[k], tmp_eval_f1[k] = Scoring_nn.flat_metrics(self, logits, label_ids, k)

        # Accumulate metircs.
        eval_accuracy += tmp_eval_accuracy

        for k in range(category_number):
          eval_precision[k] += tmp_eval_precision[k]
          eval_recall[k] += tmp_eval_recall[k]
          eval_f1[k] += tmp_eval_f1[k]

        # Track the number of batches
        nb_eval_steps += 1

        del b_input_ids, b_input_mask, b_labels, outputs, logits, label_ids
        torch.cuda.empty_cache()
        # Report the final accuracy for this validation run.
      # print(eval_accuracy)
      print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))

      # Calculate the average value for each metrics
      precision_avg, recall_avg, f1_avg = 0, 0, 0
      for k in range(category_number):
        print("Category: %d" % (k))
        eval_precision[k] = eval_precision[k] / nb_eval_steps
        precision_avg += eval_precision[k]
        eval_recall[k] = eval_recall[k] / nb_eval_steps
        recall_avg += eval_recall[k]
        eval_f1[k] = eval_f1[k] / nb_eval_steps
        f1_avg += eval_f1[k]
        print("  Precision: {0:.4f}".format(eval_precision[k]))
        print("  Recall: {0:.4f}".format(eval_recall[k]))
        print("  F1: {0:.4f}".format(eval_f1[k]))

      # Display these metrics
      print("The average precision is: {0:.4f}".format(precision_avg / category_number))
      print("The average recall is: {0:.4f}".format(recall_avg / category_number))
      print("The average f1 is: {0:.4f}".format(f1_avg / category_number))
      print("  Validation took: {:}".format(Scoring_nn.format_time(self,  time.time() - t0)))

      # Append metrics to list and store them
      val_precision_all.append(eval_precision)
      val_recall_all.append(eval_recall)
      val_f1_all.append(eval_f1)
      val_acc_all.append(eval_accuracy / nb_eval_steps)
      avg_val_f1_all.append(f1_avg / category_number)
      avg_val_recall_all.append(recall_avg / category_number)
      avg_val_precision_all.append(recall_avg / category_number)
      
      # Compare the best average f1 score and store the best model
      avg_val_f1 = f1_avg / category_number
      if avg_val_f1 > current_best_F1_avg:
        print("Found new best average F1 score model when validation, old f1 is %f , new f1 is %f" %(best_F1_avg, avg_val_f1))
        torch.save(model.state_dict(),'Best_val_Bert_rectified_priorization.pth')
        current_best_F1_avg = avg_val_f1

    print("")
    print("Training complete!")

    # Store all metrics duing training and validation. Return it
    metrics_all = defaultdict()
    metrics_all['train_loss_values'], metrics_all['train_acc_all'], metrics_all[
      'train_precision_all'] = train_loss_values, train_acc_all, train_precision_all
    metrics_all['train_recall_all'], metrics_all['train_f1_all'], metrics_all['val_acc_all'], metrics_all[
      'val_precision_all'], metrics_all['val_recall_all'], metrics_all[
      'val_f1_all'] = train_recall_all, train_f1_all, val_acc_all, val_precision_all, val_recall_all, val_f1_all
    metrics_all['avg_train_f1_all'], metrics_all['avg_train_recall_all'], metrics_all[
      ' avg_train_precision_all'] = avg_train_f1_all, avg_train_recall_all, avg_train_precision_all
    metrics_all['avg_val_f1_all'], metrics_all['avg_val_recall_all'], metrics_all[
      'avg_val_precision_all'] = avg_val_f1_all, avg_val_recall_all, avg_val_precision_all
    return metrics_all, current_best_F1_avg

  def evalTest(self, model, test_dataloader, category_number):
    '''This method is used to only validate trained BERT model on a test set'''
    # ========================================
    #               Testing
    # ========================================
    print()
    print('Testing...')
    # record the start time
    t0 = time.time()

    # Identify GPU to use
    if torch.cuda.is_available():
      device = torch.device("cuda")
      print('There are %d GPU(s) available.' % torch.cuda.device_count())
      print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
      print('No GPU available, using the CPU instead.')
      device = torch.device("cpu")
    # put model in evaluation model
    model.eval()

    # Build list to store metrics of each epoches
    test_acc_all = []
    test_precision_all = []
    test_recall_all = []
    test_f1_all = []
    test_accuracy = 0
    nb_test_steps, nb_test_examples = 0, 0

    # Build the dict for each metrics and store metrics for each category
    test_precision, test_recall, test_f1 = {}, {}, {}
    for i in range(category_number):
      test_f1[i], test_precision[i], test_recall[i] = 0, 0, 0

    # For each batch of training data...
    for batch in test_dataloader:
      # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)

      # Unpack the inputs from our dataloader
      b_input_ids, b_input_mask, b_labels = batch

      # Telling the model not to compute or store gradients, saving memory and
      # speeding up validation
      with torch.no_grad():

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
      # Get the "logits" output by the model. The "logits" are the output
      # values prior to applying an activation function like the softmax.
      logits = outputs[0]

      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()

      # Calculate the accuracy for this batch of test sentences.
      tmp_test_accuracy = Scoring_nn.flat_accuracy(self, logits, label_ids)

      tmp_test_precision = {}
      tmp_test_recall = {}
      tmp_test_f1 = {}
      for i in range(category_number):
        tmp_test_precision[i] = 0
        tmp_test_recall[i] = 0
        tmp_test_f1[i] = 0

      for k in range(category_number):
        tmp_test_precision[k], tmp_test_recall[k], tmp_test_f1[k] = Scoring_nn.flat_metrics(self, logits, label_ids, k)

      # Accumulate the total accuracy.
      test_accuracy += tmp_test_accuracy

      for k in range(category_number):
        test_precision[k] += tmp_test_precision[k]
        test_recall[k] += tmp_test_recall[k]
        test_f1[k] += tmp_test_f1[k]

      # Track the number of batches
      nb_test_steps += 1

      # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(test_accuracy / nb_test_steps))

    # Calculate the average value for each metrics
    precision_avg, recall_avg, f1_avg = 0, 0, 0
    for k in range(category_number):
      print("Category: %d" % (k))
      test_precision[k] = test_precision[k] / nb_test_steps
      precision_avg += test_precision[k]
      test_recall[k] = test_recall[k] / nb_test_steps
      recall_avg += test_recall[k]
      test_f1[k] = test_f1[k] / nb_test_steps
      f1_avg += test_f1[k]
      print("  Precision: {0:.4f}".format(test_precision[k]))
      print("  Recall: {0:.4f}".format(test_recall[k]))
      print("  F1: {0:.4f}".format(test_f1[k]))
    # Report other metrics
    print("The average precision is: {0:.4f}".format(precision_avg / category_number))
    print("The average recall is: {0:.4f}".format(recall_avg / category_number))
    print("The average f1 is: {0:.4f}".format(f1_avg / category_number))
    print("  Testing took: {:}".format(Scoring_nn.format_time(self, time.time() - t0)))

    # Store them in a list
    test_precision_all.append(test_precision)
    test_recall_all.append(test_recall)
    test_f1_all.append(test_f1)
    test_acc_all.append(test_accuracy / nb_test_steps)

    
    return {'precision': test_precision_all, 'recall': test_recall_all, 'f1': test_f1_all, 'acc': test_acc_all}


class Bertnn_info(Bertnn):
  def __init__(self,train_inputs,train_labels,train_masks, validation_inputs,
               validation_labels,validation_masks,test_batch, batch_size=16,
               epochs=4,lr=None,category_number=2):
    super().__init__(train_inputs,train_labels,train_masks, validation_inputs,
               validation_labels,validation_masks,test_batch, batch_size=16,
               epochs=4,lr=None,category_number=2,evluate_methods= Evaluate_model_nn())

class Bertnn_priorization(Bertnn):
  def __init__(self,train_inputs,train_labels,train_masks, validation_inputs,
               validation_labels,validation_masks,test_batch, batch_size=16,
               epochs=4,lr=None,category_number=2):
    categories = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}
    super().__init__(train_inputs, train_labels, train_masks, validation_inputs,
                     validation_labels, validation_masks, test_batch, batch_size=16,
                     epochs=4, lr=None, category_number=4, categories=categories,
                     evluate_methods=Evaluate_model_nn())

class Bertnn_rectified_info(Bertnn):
  def __init__(self,train_inputs,train_labels,train_masks, validation_inputs,
               validation_labels,validation_masks,test_batch, batch_size=16,
               epochs=4,lr=None,category_number=2):
    super().__init__(train_inputs,train_labels,train_masks, validation_inputs,
               validation_labels,validation_masks,test_batch, batch_size=16,
               epochs=4,lr=None,category_number=2,evluate_methods= Evaluate_model_nn_rectified_binary())

  def updateParameters(self,train_dataloader,lr=None):
    total_steps = len(train_dataloader) * self.epochs
    # If new leraning rate is passed to this method, then update it
    if lr != None:
      self.optimizer_large = AdamW(self.model.parameters(),
                lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
      self.optimizer_medium = AdamW(self.model.parameters(),
                lr = 3e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
      self.optimizer_small = AdamW(self.model.parameters(),
                lr = 1e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
      self.optimizer = AdamW(self.model.parameters(),
                lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
    else:
      self.optimizer = AdamW(self.model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
      # Create the learning rate scheduler.
      self.scheduler_large = get_linear_schedule_with_warmup(self.optimizer_large,
                            num_warmup_steps = 0, # Default value in run_glue.py
                            num_training_steps = total_steps)
      self.scheduler_medium = get_linear_schedule_with_warmup(self.optimizer_medium,
                            num_warmup_steps = 0, # Default value in run_glue.py
                            num_training_steps = total_steps*3)
      self.scheduler_small = get_linear_schedule_with_warmup(self.optimizer_small,
                            num_warmup_steps = 0, # Default value in run_glue.py
                            num_training_steps = total_steps/2)
      self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                            num_warmup_steps = 0, # Default value in run_glue.py
                            num_training_steps = total_steps)

  def searchUpsample(self, increaseTimes,lr=None):
    '''This method is used to try different over-sample factor to train BERT model'''
    # build dict and list to store metrics
    all_metrics = {}  # all metrics
    all_test_metrics = {}  # all test metrcis
    bestF1 = defaultdict()  # store the best F1 score
    bestF1['bestScore'] = 0
    bestF1['bestIncreaseTime'] = None  # store the best over-sample factor
    for increaseTime in range(0, increaseTimes, 10):

      print('Current increase %d times' % (increaseTime))
      # protect the original dataset
      train_inputs_upsample = list(copy.deepcopy(self.train_inputs))
      train_labels_upsample = list(copy.deepcopy(self.train_labels))
      train_masks_upsample = list(copy.deepcopy(self.train_masks))

      # up-sample the label for each category
      for category in range(self.category_number):
        # Does not up-sample the majority class
        if category == 0:
          continue
        print("Current is processing category %d" % (category))
        # build the up-sample data set
        train_inputs_upsample, train_labels_upsample, train_masks_upsample = Up_sample_Bertnn.upSample(train_inputs_upsample,
                                                                                           train_labels_upsample,
                                                                                           train_masks_upsample,
                                                                                           category, increaseTime)
        print()

      # Repost stastistic of this category before and after over=sampling
      for category in range(self.category_number):
        print('Currrent is processing on categorie %s' % (self.categories[category]))
        print()

        print('Before up-sample, the ratio of current category and all samples')
        print(np.count_nonzero(self.train_labels == category))
        print(np.count_nonzero(self.train_labels == category) / len(self.train_labels))
        print()

        print('After up-sample, the ratio of current category and all samples')
        print(np.count_nonzero(np.array(train_labels_upsample) == category))
        print(np.count_nonzero(np.array(train_labels_upsample) == category) / len(train_labels_upsample))
        print()

      # Use up-sample data set to train nerual network

      train_inputs = torch.tensor(np.array(train_inputs_upsample)).to(torch.int64)
      train_labels = torch.tensor(np.array(train_labels_upsample)).to(torch.int64)
      train_masks = torch.tensor(np.array(train_masks_upsample)).to(torch.int64)

      # Create the DataLoader for our training set.
      train_data = TensorDataset(train_inputs, train_masks, train_labels)
      train_dataloader = self.createDataloader(train_data)

      # Number of training epochs (authors recommend between 2 and 4)
      self.epochs = 4

      # Re-initilise the Bert model
      self.iniModel()
      if self.lr != None:
        self.updateParameters(train_dataloader, lr)
      else:
        self.updateParameters(train_dataloader)
      # Use over-sample dataset to train and validate BERT model
      all_metrics[increaseTime], val_F1_avg = self.evluate_methods.evalTrain(self.model, train_dataloader, self.validation_dataloader, self.epochs,
                self.category_number, self.optimizer, self.scheduler,
                self.optimizer_small,  self.scheduler_small, )
      # Store return metrics result
      # all_test_metrics[increaseTime], test_F1_avg = self.evluate_methods.evalTest()
      if val_F1_avg > best_F1_avg:
        print('Best increase factor is %d' %(increaseTime))
        best_F1_avg = val_F1_avg 
              
      # Delete over-sample dataset
      del self.model, train_inputs, train_labels, train_masks, train_data, train_dataloader

    # Delete BERT mdoel
    self.terminate()
    return all_metrics, all_test_metrics

  def upSampleTrain(self, increaseTime):
    '''This method is design to use a over-sample method to train BERT model and evaluate it'''

    print('Current increase %d times to train' % (increaseTime))
    # protect the original dataset
    # protect the original dataset
    train_inputs_upsample = copy.deepcopy(self.train_inputs)
    train_labels_upsample = copy.deepcopy(self.train_labels)
    train_masks_upsample = copy.deepcopy(self.train_masks)
    # up-sample the label for each category
    for category in range(4):
      print("Current is processing category %d" % (category))
      # build the up-sample data set
      train_inputs_upsample, train_labels_upsample, train_masks_upsample = Up_sample_Bertnn.upSample(train_inputs_upsample,
                                                                                         train_labels_upsample,
                                                                                         train_masks_upsample, category,
                                                                                         increaseTime)
      print()



    # Use up-sample data set to train nerual network
    print(np.array(train_inputs_upsample).shape)
    train_inputs = torch.tensor(np.array(train_inputs_upsample)).to(torch.int64)
    train_labels = torch.tensor(np.array(train_labels_upsample)).to(torch.int64)
    train_masks = torch.tensor(np.array(train_masks_upsample)).to(torch.int64)

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_dataloader = self.createDataloader(train_data)

    # Number of training epochs (authors recommend between 2 and 4)
    self.epochs = 2

    # Re-initilise the Bert model
    self.iniModel()
    # if new learning rate is passed to this method, then udpate the learning rate
    if self.lr != None:
      self.updateParameters(train_dataloader, self.lr)
    else:
      self.updateParameters(train_dataloader)

    return self.evluate_methods.evalTrain(self.model, train_dataloader, self.validation_dataloader, self.epochs,
                self.category_number, self.optimizer, self.scheduler,
                self.optimizer_small, self.scheduler_small, )

class Bertnn_rectified_priorization(Bertnn):
  def __init__(self,train_inputs,train_labels,train_masks, validation_inputs,
               validation_labels,validation_masks,test_batch, batch_size=16,
               epochs=4,lr=None,category_number=4):
    categories = {0:'Low',1:'Medium',2:'High',3:'Critical'}
    super().__init__(train_inputs, train_labels, train_masks, validation_inputs,
                     validation_labels, validation_masks, test_batch, batch_size=16,
                     epochs=4, lr=None, category_number=4, categories=categories,
                     evluate_methods=Evaluate_model_nn_rectified_multi_classes())

  def updateParameters(self,train_dataloader,lr=None):
    total_steps = len(train_dataloader) * self.epochs
    # If new leraning rate is passed to this method, then update it
    if lr != None:
      self.optimizer_large = AdamW(self.model.parameters(),
                lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
      self.optimizer_medium = AdamW(self.model.parameters(),
                lr = 3e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
      self.optimizer_small = AdamW(self.model.parameters(),
                lr = 1e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
      self.optimizer = AdamW(self.model.parameters(),
                lr = lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
    else:
      self.optimizer_large = AdamW(self.model.parameters(),
                lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
      self.optimizer_medium = AdamW(self.model.parameters(),
                lr = 3e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
      self.optimizer_small = AdamW(self.model.parameters(),
                lr = 1e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
      self.optimizer = AdamW(self.model.parameters(),
                lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
    # Create the learning rate scheduler.
    self.scheduler_large = get_linear_schedule_with_warmup(self.optimizer_large,
                          num_warmup_steps = 0, # Default value in run_glue.py
                          num_training_steps = total_steps)
    self.scheduler_medium = get_linear_schedule_with_warmup(self.optimizer_medium,
                          num_warmup_steps = 0, # Default value in run_glue.py
                          num_training_steps = total_steps*3)
    self.scheduler_small = get_linear_schedule_with_warmup(self.optimizer_small,
                          num_warmup_steps = 0, # Default value in run_glue.py
                          num_training_steps = total_steps/2)
    self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                          num_warmup_steps = 0, # Default value in run_glue.py
                          num_training_steps = total_steps)

  def searchUpsample(self, increaseTimes,lr=None):
    '''This method is used to try different over-sample factor to train BERT model'''
    # build dict and list to store metrics
    all_metrics = {}  # all metrics
    all_test_metrics = {}  # all test metrcis
    bestF1 = defaultdict()  # store the best F1 score
    bestF1['bestScore'] = 0
    bestF1['bestIncreaseTime'] = None  # store the best over-sample factor
    best_model_wts = self.model.state_dict()
    best_F1_avg = 0

    for increaseTime in range(0, increaseTimes, 10):

      print('Current increase %d times' % (increaseTime))
      # protect the original dataset
      train_inputs_upsample = list(copy.deepcopy(self.train_inputs))
      train_labels_upsample = list(copy.deepcopy(self.train_labels))
      train_masks_upsample = list(copy.deepcopy(self.train_masks))

      # up-sample the label for each category
      for category in range(self.category_number):
        # Does not up-sample the majority class
        if category == 0:
          continue
        print("Current is processing category %d" % (category))
        # build the up-sample data set
        train_inputs_upsample, train_labels_upsample, train_masks_upsample = Up_sample_Bertnn.upSample(self, train_inputs_upsample,
                                                                                           train_labels_upsample,
                                                                                           train_masks_upsample,
                                                                                           category, increaseTime)
        print()

      # Repost stastistic of this category before and after over=sampling
      for category in range(self.category_number):
        print('Currrent is processing on categorie %s' % (self.categories[category]))
        print()

        print('Before up-sample, the ratio of current category and all samples')
        print(np.count_nonzero(self.train_labels == category))
        print(np.count_nonzero(self.train_labels == category) / len(self.train_labels))
        print()

        print('After up-sample, the ratio of current category and all samples')
        print(np.count_nonzero(np.array(train_labels_upsample) == category))
        print(np.count_nonzero(np.array(train_labels_upsample) == category) / len(train_labels_upsample))
        print()

      # Use up-sample data set to train nerual network

      train_inputs = torch.tensor(np.array(train_inputs_upsample)).to(torch.int64)
      train_labels = torch.tensor(np.array(train_labels_upsample)).to(torch.int64)
      train_masks = torch.tensor(np.array(train_masks_upsample)).to(torch.int64)

      # Create the DataLoader for our training set.
      train_data = TensorDataset(train_inputs, train_masks, train_labels)
      train_dataloader = self.createDataloader(train_data)

      # Number of training epochs (authors recommend between 2 and 4)
      self.epochs = 4

      # Re-initilise the Bert model
      self.iniModel()
      if self.lr != None:
        self.updateParameters(train_dataloader, lr)
      else:
        self.updateParameters(train_dataloader)
      # Use over-sample dataset to train and validate BERT model
      all_metrics[increaseTime], val_F1_avg  = self.evluate_methods.evalTrain(self.model, train_dataloader, self.validation_dataloader, self.epochs,
                self.category_number, self.optimizer, self.scheduler,
                self.optimizer_small, self.optimizer_medium, self.scheduler_small, self.scheduler_medium, best_F1_avg = best_F1_avg)
      # Store return metrics result
      # all_test_metrics[increaseTime], test_F1_avg = self.evluate_methods.evalTest(self, best_F1_avg = best_F1_avg)
      if val_F1_avg > best_F1_avg:
        print('Best increase factor is %d' %(increaseTime))
        best_F1_avg = val_F1_avg 

      # Delete over-sample dataset
      del self.model, train_inputs, train_labels, train_masks, train_data, train_dataloader

    # # Save the best model
    # torch.save(self.model.state_dict(),'Best_Bert_rectified_priorization.pth')

    # Delete BERT mdoel
    self.terminate()
    return all_metrics, all_test_metrics

  def upSampleTrain(self, increaseTime):
    '''This method is design to use a over-sample method to train BERT model and evaluate it'''

    print('Current increase %d times to train' % (increaseTime))
    # protect the original dataset
    # protect the original dataset
    train_inputs_upsample = copy.deepcopy(self.train_inputs)
    train_labels_upsample = copy.deepcopy(self.train_labels)
    train_masks_upsample = copy.deepcopy(self.train_masks)
    # up-sample the label for each category
    for category in range(4):
      print("Current is processing category %d" % (category))
      # build the up-sample data set
      train_inputs_upsample, train_labels_upsample, train_masks_upsample = Up_sample_Bertnn.upSample(train_inputs_upsample,
                                                                                         train_labels_upsample,
                                                                                         train_masks_upsample, category,
                                                                                         increaseTime)
      print()



    # Use up-sample data set to train nerual network
    print(np.array(train_inputs_upsample).shape)
    train_inputs = torch.tensor(np.array(train_inputs_upsample)).to(torch.int64)
    train_labels = torch.tensor(np.array(train_labels_upsample)).to(torch.int64)
    train_masks = torch.tensor(np.array(train_masks_upsample)).to(torch.int64)

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_dataloader = self.createDataloader(train_data)

    # Number of training epochs (authors recommend between 2 and 4)
    self.epochs = 2

    # Re-initilise the Bert model
    self.iniModel()
    # if new learning rate is passed to this method, then udpate the learning rate
    if self.lr != None:
      self.updateParameters(train_dataloader, self.lr)
    else:
      self.updateParameters(train_dataloader)

    return self.evluate_methods.evalTrain(self.model, train_dataloader, self.validation_dataloader, self.epochs,
                self.category_number, self.optimizer, self.scheduler,
                self.optimizer_small, self.optimizer_medium, self.scheduler_small, self.scheduler_medium)


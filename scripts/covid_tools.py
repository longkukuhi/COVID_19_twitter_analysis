from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import datetime, time
from sklearn.metrics import fbeta_score
from sklearn.model_selection import cross_validate, RandomizedSearchCV, KFold
from collections import defaultdict
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import seaborn as sns
import json
import re
import spacy
from spacy.tokenizer import Tokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.tree import DecisionTreeClassifier
import copy
from sklearn.metrics import plot_confusion_matrix


################
# Functions and class coded by Zijun Long for classifying Covid-19 related tweets
# Email: 2452593L@student.gla.ac.uk
# Version: 3 Update date: 02/02/2021
################

class Scoring():
  
  @classmethod
  def format_time(cls, elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


class Up_sample():
  @classmethod
  def over_sample_from_frame(cls, features_frame, labels_frame, categories, increase_factor):
    trainSet_temp = copy.deepcopy(features_frame)
    label_temp = copy.deepcopy(labels_frame)
    features_list = trainSet_temp.keys()

    for category in categories:
      idxs = np.where(np.array(label_temp) == category)[0]
      # iterate all sample belong to current processing category and duplicate all features
      for idx in list(idxs):
        # Duplicate (increaseTime) times
        for j in range(increase_factor):
          for key in features_list:
            trainSet_temp[key].append(trainSet_temp[key][idx])
          label_temp.append(category)

    return trainSet_temp, label_temp

  @classmethod
  def up_sample(cls, location, category, increaseTime, testF, tokenizer):
    '''
    This function build up-sample dataset from csv file
    :param location: name of processing location
    :param category: indicate current processing category
    :param increaseTime: over-sample factor
    :param testF: test features
    :param tokenizer: tokenizer to tokenize
    :return: up-sample training set, up-sample training labels, transformed test features
    '''
    import copy
    # Build up-sample

    df_tmp = pd.read_csv('cache/' + location + '/upsampleData_' + str(category) + '_' + str(increaseTime) + '.csv')
    featureList = ['full_text', 'hashtags', 'favorite_count']
    trainSet_temp = Process_dataset.extract_features(df_tmp, featureList)
    labelList = ['target']
    train_label_temp = Process_dataset.extract_features(df_tmp, labelList)['target']


    # featureUnion multiple features
    Tfidf_vect = TfidfVectorizer(tokenizer=tokenizer, max_features=25000, ngram_range=(1, 2),
                                 sublinear_tf=(1, 2))
    count_vect = CountVectorizer(tokenizer=tokenizer)

    fu_temp = FeatureUnion([('full_text', Pipeline([('selector', ItemSelector(key='full_text')), ('tf_idf', Tfidf_vect),
                                                    ])),
                            ('hashtags', Pipeline([('selector', ItemSelector(key='hashtags')),
                                                   ('cv', count_vect),
                                                   ])),
                            ('favorite_count', Pipeline([('selector', numericalTransformer(key='favorite_count')),
                                                         ])),
                            ], n_jobs=-1)

    trainSet_temp = fu_temp.fit_transform(trainSet_temp)
    testF_tmp = fu_temp.transform(testF)
    return trainSet_temp, train_label_temp, testF_tmp
  
  @classmethod
  def build_upsample_set(cls, trainSet, trainL_single, location, category, increaseTime):
    '''
    This function use over-sample method to build over-sampling
    data set and store them into csv file
    :param trainSet: training features
    :param trainL_single: A dict training labels for every categories
    :param location: current processing location
    :param category: current processing category
    :param increaseTime: maximum over-sample factor
    :return: None
    '''
    # Build up-sample
    import copy
    print('Current increase %d times' % (increaseTime))
    # Protect original data set

    trainSet_temp = copy.deepcopy(trainSet)
    label_temp = copy.deepcopy(trainL_single)[category]

    # Find all sample belong to current processing category
    # print('Before length %d'%(len(label_temp)))
    idxs = np.where(np.array(label_temp) == 1)[0]
    # print('There are %d postitive sample in this category.'%(len(idxs)))

    # iterate all sample belong to current processing category and duplicate all features
    for idx in list(idxs):
      # Duplicate (increaseTime) times
      if category == 8 or category == 3:
        break
      for j in range(increaseTime):
        trainSet_temp['hashtags'].append(trainSet_temp['hashtags'][idx])
        trainSet_temp['full_text'].append(trainSet_temp['full_text'][idx])
        trainSet_temp['favorite_count'].append(trainSet_temp['favorite_count'][idx])
        label_temp.append(1)


    # Store over-sampling data set into csv file
    df_tmp = pd.DataFrame({'full_text': trainSet_temp['full_text'], 'hashtags': trainSet_temp['hashtags'],
                           'favorite_count': trainSet_temp['favorite_count'], 'target': label_temp})
    df_tmp.to_csv('cache/' + location + '/upsampleData_' + str(category) + '_' + str(increaseTime) + '.csv',
                  index=False)

  @classmethod
  def search_best_lR(cls, increaseTimes, tokenizer, testF, testL):
    '''
    This function search the best over-sample factor for classifier
    :param increaseTimes: over-sample factor
    :param tokenizer: tonizer to tokenize
    :param testF: test features
    :param testL: test label
    :return: metrics of all over-sample factor,
            best parameters of over-sample factor,
             cross validation score of over-sample factor
    '''
    categories = ['GoodsServices', 'InformationWanted', 'Volunteer', 'MovePeople', 'EmergingThreats', 'NewSubEvent',
                  'ServiceAvailable', 'Advice', 'Any']
    count_vect = CountVectorizer(tokenizer=tokenizer)
    Tfidf_vect = TfidfVectorizer(tokenizer=tokenizer)

    # iterate over a range of increaseTime and all category

    # store the best parameter for each increaseTime
    metrics_all = {'f1_best_all': defaultdict(), 'best_paras': defaultdict()}
    best_parameters_all = defaultdict()
    cv_score_all = defaultdict()
    test_scores = defaultdict()
    for increaseTime in range(0, increaseTimes):
      metrics_one_run = {'f1_best_each_cate': defaultdict()}
      best_parameters_one_run = defaultdict()
      cv_score_one_run = defaultdict()
      # To accerate this searching, only try one value for each category
      print("Performing grid search on IncreaseTime %d" % (increaseTime))
      for category in range(9):
        if category == 1 and increaseTime != 6:
          continue
        if category == 2 and increaseTime != 5:
          continue
        if category == 4 and increaseTime != 6:
          continue
        if category == 5 and increaseTime != 5:
          continue
        if category == 6 and increaseTime != 6:
          continue
        if category == 7 and increaseTime != 4:
          continue
        if category == 3 or category == 8:
          continue

        print("Performing grid search on category %s" % (categories[category]))
        df_tmp = pd.read_csv('cache/upsampleData_' + str(category) + '_' + str(increaseTime) + '.csv')
        trainF = {'full_text': df_tmp['full_text'], 'hashtags': df_tmp['hashtags'],
                  'favorite_count': df_tmp['favorite_count']}
        trainL = list(df_tmp['target'])

        # build the training set for feature union and gridSerachCV
        trainF_tmp = []
        for i in range(len(trainF['full_text'])):
          trainF_tmp.append({'full_text': trainF['full_text'][i], 'hashtags': trainF['hashtags'][i],
                             'favorite_count': trainF['favorite_count'][i]})

        fu = FeatureUnion(
          [('full_text', Pipeline([('selector', ItemSelector(key='full_text')), ('tf_idf', Tfidf_vect),
                                   ])),
           ('hashtags', Pipeline([('selector', ItemSelector(key='hashtags')),
                                  ('cv', count_vect),
                                  ])),
           ('favorite_count', Pipeline([('selector', numericalTransformer(key='favorite_count')),
                                        ])),
           ])
        fullpipe = Pipeline([

          ('fu', fu),
          ('lr', LogisticRegression())
        ])

        params = {
          # Fill in the parameter

          'fu__full_text__tf_idf__ngram_range': [(1, 2), (1, 3), (1, 4)],
          'fu__full_text__tf_idf__sublinear_tf': (True, False),
          'fu__full_text__tf_idf__max_features': np.linspace(1000, 50000, 100, dtype=int),
          'lr__penalty': ['l2'],
          'lr__C': np.linspace(0.000001, 1000, 100, dtype=float),
          'lr__class_weight': ['balanced', 'None'],
          'lr__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
          'lr__max_iter': np.linspace(50, 10000, 100, dtype=int),
        }

        # Pass in your pipeline, params (to tune), scoring, and split parameters here!
        grid_search = RandomizedSearchCV(fullpipe, params, verbose=1, cv=3, n_iter=500, scoring='f1', n_jobs=-1,
                                         pre_dispatch='2*n_jobs', )

        # print("pipeline:", [name for name, _ in pipe.steps])
        # print("parameters:")
        # print(params)
        # n_jobs=-1,pre_dispatch='2*n_jobs'
        # FILL IN HERE -- Fit grid_search on the combined train/validation data and labels.
        grid_search.fit(trainF_tmp, trainL)

        # evaluate the test result
        testF_union_tmp = grid_search.transform(testF)

        predictions = grid_search.predict(testF_union_tmp)
        precision = precision_score(predictions, testL[category])
        recall = recall_score(predictions, testL[category])
        accuracy = accuracy_score(predictions, testL[category])
        f1 = fbeta_score(predictions, testL[category], 1, )

        test_scores = {'accuracy': accuracy, 'precision': precision, 'f1': f1, 'recall': recall}
        print("Test result:", test_scores)

        print('type')
        print(type(trainL[0]))
        print("Best score: %0.3f" % grid_search.best_score_)
        # print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        parameters_dict = {}
        for param_name in sorted(params.keys()):
          print("\t%s: %r" % (param_name, best_parameters[param_name]))
          parameters_dict[param_name] = best_parameters[param_name]

        metrics_one_run['f1_best_each_cate'][category] = grid_search.best_score_
        cv_score_one_run[category] = grid_search.cv_results_
        best_parameters_one_run[category] = parameters_dict
        print(best_parameters_one_run[category])
        test_scores[increaseTime][category] = test_scores

      best_parameters_all[increaseTime] = best_parameters_one_run
      metrics_all['f1_best_all'][increaseTime] = metrics_one_run
      cv_score_all[increaseTime] = best_parameters_one_run
    return metrics_all, best_parameters_all, cv_score_all

class Feature_union_pipeline():
  @classmethod
  def build_fu_from_feature_list(cls, full_text_vectorizer, hash_tags_vectorizer):
    fu = FeatureUnion(
      [('full_text', myPipeline([('selector', ItemSelector(key='full_text')), ('cv', full_text_vectorizer),
                                 ])),
       ('hashtags', myPipeline([('selector', ItemSelector(key='hashtags')),
                                ('cv', hash_tags_vectorizer),
                                ])),
       ('favorite_count',myPipeline([('selector', numericalTransformer(key='favorite_count')),
       ])),
       ], n_jobs=-1)
    return fu

class Evaluate_model():
  @classmethod
  def evaluate_prediction(cls, predictions, true_lables, detail_sign = True):
    type_number = len(set(true_lables))
    if type_number <= 2:
      precision = precision_score(predictions, true_lables)
      recall = recall_score(predictions, true_lables)
      accuracy = accuracy_score(predictions, true_lables)
      f1 = fbeta_score(predictions, true_lables, 1, )
    else:
      precision = precision_score(predictions, true_lables, average='macro')
      recall = recall_score(predictions, true_lables, average='macro')
      accuracy = accuracy_score(predictions, true_lables)
      f1 = fbeta_score(predictions, true_lables, 1, average='macro')

    test_scores = {'accuracy': accuracy, 'precision': precision, 'f1': f1, 'recall': recall}
    print("Prediction result:", test_scores)

    if detail_sign == True:
      print(classification_report(predictions, true_lables, digits=4))

  @classmethod
  def evaluation_summary(cls, description, predictions, true_labels):
    '''
    This function conduct total evaluation on given prediction and true labels
    :param description: text description of current classifier
    :param predictions: prediction of classifier
    :param true_labels: true labels
    :return: precision, recall, f1 score
    '''
    print("Evaluation for: " + description)

    label_number = len(set(true_labels))
    # choose metrics based on different number of categories
    if label_number > 2:
      precision = precision_score(predictions, true_labels, average='macro')
      recall = recall_score(predictions, true_labels, average='macro')
      accuracy = accuracy_score(predictions, true_labels)
      f1 = fbeta_score(predictions, true_labels, 1, average='macro')
    else:
      precision = precision_score(predictions, true_labels, average='binary')
      recall = recall_score(predictions, true_labels, average='binary')
      accuracy = accuracy_score(predictions, true_labels)
      f1 = fbeta_score(predictions, true_labels, 1, average='binary')

    print("Classifier '%s' has Acc=%0.3f P=%0.3f R=%0.3f F1=%0.3f"
          % (description, accuracy, precision, recall, f1))
    # Specify three digits instead of the default two.
    print(classification_report(predictions, true_labels, digits=3))
    return precision, recall, f1

  @classmethod
  def cross_validation_result(cls, cf, dataset, labels):
    '''
    This function run cross validation on given dataset and return average score
    :param cf: classifier to evaluate
    :param dataset: dataset used to eavluate
    :param labels: true labels
    :return: average cross validation result
    '''
    # Cross validation
    label_number = len(set(labels))
    # choose metrics based on different number of categories
    if label_number > 2:
      # run cross validation
      scores = cross_validate(cf, dataset, labels, cv=5,
                              scoring=['accuracy', 'precision_macro', 'f1_macro', 'recall_macro'],
                              return_train_score=True)
    else:
      scores = cross_validate(cf, dataset, labels, cv=5,
                              scoring=['accuracy', 'precision', 'f1', 'recall'], return_train_score=True)
    # calculate the average value for each metric
    for key in scores.keys():
      scores[key] = scores[key].mean()

    # print cv_scores
    cv_scores = scores
    print('Cross validation: Average training set result: %s=%0.6f %s=%0.6f %s=%0.6f %s=%0.6f .'
          % (list(cv_scores.keys())[2], list(cv_scores.values())[2],
             list(cv_scores.keys())[4], list(cv_scores.values())[4],
             list(cv_scores.keys())[6], list(cv_scores.values())[6],
             list(cv_scores.keys())[8], list(cv_scores.values())[8]))
    print()

    print('Cross validation: Average validation set result: %s=%0.6f %s=%0.6f %s=%0.6f %s=%0.6f .'
          % (list(cv_scores.keys())[3], list(cv_scores.values())[3],
             list(cv_scores.keys())[5], list(cv_scores.values())[5],
             list(cv_scores.keys())[7], list(cv_scores.values())[7],
             list(cv_scores.keys())[9], list(cv_scores.values())[9]))
    print()
    return scores



  @classmethod
  def total_evaluation(cls, model, model_name, runningSet,
                       print_out_detail=False,
                       ):
    '''
    Conduct total evaluation on given dataset and model
    :param model: model to evalutae
    :param model_name: text, name of model
    :param runningSet: A list contrains training features, training labels
                          test features, test labels
    :param print_out_detail: a sign whether print out verbose result
    :param multi_sign: a sign whether the number of categories is over 2
    :return:
    '''
    train_set, train_labels = runningSet[:2]
    test_set, test_labels = runningSet[2:4]

    type_number = len(set(train_labels))
    from sklearn.model_selection import cross_validate
    # Cross validation
    cv_scores = Evaluate_model.cross_validation_result(model, train_set, train_labels)

    print('Evaluating on test set:')
    model.fit(train_set, train_labels)
    test_predicted_labels = model.predict(test_set)
    precision_test, recall_test, f1_test = Evaluate_model.evaluation_summary(model_name + "result on test set",
                                                                     test_predicted_labels, test_labels)

    print('Simple test result:')
    Evaluate_model.evaluate_prediction(test_predicted_labels, test_labels)

    # detial result for analysis
    if print_out_detail == True:
      # Detail cross-validation result
      print('Detail cross-validation result:')
      #     train_set = np.array(train_set)
      train_labels = np.array(train_labels)
      kf = KFold(n_splits=5)
      for train_index, test_index in kf.split(train_set):
        x_train, x_test = train_set[train_index], train_set[test_index]
        #         print(train_index)
        y_train, y_test = train_labels[train_index], train_labels[test_index]

        # evaluation_summary
        model.fit(x_train, y_train)
        train_predicted_labels = model.predict(x_train)
        test_predicted_labels = model.predict(x_test)

        precision_train, recall_train, f1_train = Evaluate_model.evaluation_summary(model_name + "result on training set",
                                                                     train_predicted_labels, y_train)

        precision_test, recall_test, f1_test = Evaluate_model.evaluation_summary(model_name + "result on validation set",
                                                                  test_predicted_labels, y_test)

    return cv_scores, model

  @classmethod
  def display_result_without_upsample(cls, cf, category, runningSet):
    '''
    This function print out metrcis of classifier without appling over-sampling
    :param cf: classifier to evaluate
    :param category: current processing category
    :param runningSet: A list contrains training features, training labels
                      test features, test labels
    :return: metrics on test set
    '''
    model_name = "LR result on Before up-sampling with feature union"
    # unpack running set
    train_set = runningSet[0]
    train_labels = runningSet[1]

    testL_union_tmp = runningSet[2]
    testL = runningSet[3]

    # fit the model
    cf.fit(train_set, train_labels)
    # use evalaution method to evalute the performance of classifier
    Evaluate_model.total_evaluation(None, cf, model_name, runningSet)

    # evaluate on the test set
    predictions = cf.predict(testL_union_tmp)
    type_number = len(set(testL))
    if type_number <= 2:
      precision = precision_score(predictions, testL)
      recall = recall_score(predictions, testL)
      accuracy = accuracy_score(predictions, testL)
      f1 = fbeta_score(predictions, testL, 1, )
    else:
      precision = precision_score(predictions, testL, average='macro')
      recall = recall_score(predictions, testL, average='macro')
      accuracy = accuracy_score(predictions, testL)
      f1 = fbeta_score(predictions, testL, 1, average='macro')
    # store test metrics on a dict
    test_scores = {'accuracy': accuracy, 'precision': precision, 'f1': f1, 'recall': recall}
    print("Test result:", test_scores)
    print(classification_report(predictions, testL, digits=4))
    return test_scores

  @classmethod
  def evaluate_one_cf(cls, cf, runningSet, category):
    '''
    This function evaluate the given classifier
    :param cf: classifier to evaluate
    :param runningSet: A list contrains training features, training labels
                        test features, test labels
    :param category: current processing category
    :return: corss validation score, test score
    '''
    from sklearn.linear_model import LogisticRegression

    train_set = runningSet[0]
    train_labels = runningSet[1]
    testL_union_tmp = runningSet[2]
    testL = runningSet[3]

    # LogisticRegression

    lr = cf
    # lr.fit(train_set,train_labels)
    type_number = len(set(testL))
    # choose metrics based on different number of categories
    if type_number <= 2:
      cv_scores = Evaluate_model.total_evaluation(lr, 'LogisticRegression', runningSet)

      lr.fit(train_set, train_labels)
      predictions = lr.predict(testL_union_tmp)

      precision = precision_score(predictions, testL)
      recall = recall_score(predictions, testL)
      accuracy = accuracy_score(predictions, testL)
      f1 = fbeta_score(predictions, testL, 1, )
    else:
      cv_scores = Evaluate_model.total_evaluation(lr, 'LogisticRegression', runningSet)

      lr.fit(train_set, train_labels)
      predictions = lr.predict(testL_union_tmp)

      precision = precision_score(predictions, testL, average='macro')
      recall = recall_score(predictions, testL, average='macro')
      accuracy = accuracy_score(predictions, testL)
      f1 = fbeta_score(predictions, testL, 1, average='macro')

    test_scores = {'accuracy': accuracy, 'precision': precision, 'f1': f1, 'recall': recall}
    print("Test result:", test_scores)
    print(classification_report(predictions, testL, digits=3))
    return cv_scores, test_scores

  @classmethod
  def get_errors(cls, labels, predictions, data):
    '''
    This function build a new dataFrame which contains all information of errors

    :param labels: true labels
    :param predictions: prediction of classifier
    :param data:  dataFrame contains all labelled tweets
    :return: new dataFrame which contains all information of errors
    '''
    errors = {'index': [], 'true_label': [], 'prediction': [], 'full_text': [], 'hashtags': [], 'favorite_count': []}
    label_arr = labels
    previous_text = None
    for idx, prediction in enumerate(predictions):
      label = label_arr[idx]
      if prediction != label and data['full_text'][idx] != previous_text:
        previous_text = data['full_text'][idx]
        errors['index'].append(idx)
        errors['true_label'].append(label)
        errors['prediction'].append(prediction)
        errors['full_text'].append(data['full_text'][idx])
        errors['hashtags'].append(data['hashtags'][idx])
        errors['favorite_count'].append(data['favorite_count'][idx])

    errors = pd.DataFrame(errors)
    return errors

  @classmethod
  def get_errors_indexs(cls, labels, predictions, data):
    '''
    This function return indexs of error
    :param labels: true labels
    :param predictions: prediction of classifier
    :param data:  dataFrame contains all labelled tweets
    :return: a list contains all indexs
    '''
    error_indexs = []
    previous_text = None
    for idx, prediction in enumerate(predictions):
      label = labels[idx]
      if prediction != label and data['full_text'][idx] != previous_text:
        previous_text = data['full_text'][idx]
        error_indexs.append(idx)
    return error_indexs


class Process_labels():
  def build_all_labels(self, ):
    '''
    This function read labelled tweets from five json files and
    and merge them into one dataFrame
    :return:
    '''
    import json
    with open('.\Pythonbooks\Data\COVID\label1.json') as f:
      labels1 = json.load(f)

    with open('.\Pythonbooks\Data\COVID\label2.json') as f:
      labels2 = json.load(f)

    with open('.\Pythonbooks\Data\COVID\label3.json') as f:
      labels3 = json.load(f)

    with open('.\Pythonbooks\Data\COVID\label4.json') as f:
      labels4 = json.load(f)

    with open('.\Pythonbooks\Data\COVID\label5.json') as f:
      labels5 = json.load(f)

    with open('.\Pythonbooks\Data\COVID\label6.json') as f:
      labels6 = json.load(f)

    labels11 = pd.DataFrame(labels1['events'][0]['tweets'][:])
    labels12 = pd.DataFrame(labels1['events'][1]['tweets'][:])
    labels13 = pd.DataFrame(labels1['events'][2]['tweets'][:])

    labels21 = pd.DataFrame(labels2['events'][2]['tweets'][:])
    labels22 = pd.DataFrame(labels2['events'][3]['tweets'][:])

    labels31 = pd.DataFrame(labels3['events'][0]['tweets'][:])
    # labels32 = pd.DataFrame(labels3['events'][2]['tweets'][:])

    labels41 = pd.DataFrame(labels4['events'][2]['tweets'][:])
    labels42 = pd.DataFrame(labels4['events'][3]['tweets'][:])
    labels43 = pd.DataFrame(labels4['events'][4]['tweets'][:])

    labels51 = pd.DataFrame(labels5['events'][0]['tweets'][:])
    labels52 = pd.DataFrame(labels5['events'][1]['tweets'][:])

    labels61 = pd.DataFrame(labels6['events'][0]['tweets'][:])

    # Union them all into one dataFrame
    temp = [labels11, labels12, labels13, labels21, labels22, labels31, labels41, labels42, labels43, labels51,
            labels52, labels61]
    all_labels = pd.concat(temp)
    all_labels = all_labels.rename(columns={'postID': 'id'})
    all_labels = all_labels.astype({"id": np.int64})
    all_labels.reset_index()
    return all_labels

  def build_union_labels(self, all_labels):
    '''
    This function find all labels(lines in dataFrame) for one tweet
    and union all labels, then store them into a new dataFrame without duplication
    :param all_labels: A dataFrame contains all labelled tweets
    :return: None
    '''
    # Store union lables in a list, every samples in list is a dict
    all_labels_union = []
    # For debugging
    count = 0
    # Iterate all tweets including duplicate labels
    for id in all_labels.id.values:
      count += 1
      print(count)
      priority = []
      categories = []
      # concatenate all labels for information type and prioritization task
      for idx in range(len(all_labels)):
        if id == all_labels.iloc[idx].id:
          priority += [all_labels.iloc[idx].priority]
          categories += all_labels.iloc[idx].categories

      print(list(set(priority)))
      all_labels_union.append({'id': id, 'priority': list(set(priority)), 'categories': list(set(categories))})
    # Transer list to dataFrame
    all_labels_union = pd.DataFrame(all_labels_union)
    all_labels_union.to_csv('cache/all_labeled_duplicated.csv')
    all_labels_union.drop_duplicates('id', keep='first', inplace=True)
    all_labels_union.to_csv('cache/all_labeled.csv')

    return all_labels_union

  @classmethod
  def extractLabels(cls, data, labels):
    '''
    This function transfer text label of information type task
     into numberical labels
    :param data: dataFrame contains all information of all tweets
    :param labels: A lsit contains labels of information type tasks for all tweets
    :return:
    '''
    categories = ['GoodsServices', 'InformationWanted', 'Volunteer', 'EmergingThreats', 'NewSubEvent',
                  'ServiceAvailable', 'Advice']
    labels_all = []
    # Iterate all tweets
    for i in range(len(data['full_text'])):
      label_oneTweets = [0] * 7
      # print(labels[i])
      # compare with target categories and store result
      # in number as format like [0,0,0,1,0,0,0]
      for k in range(len(labels[i])):
        for j in range(len(categories)):
          # print(labels[i][k])
          if (labels[i][k] == categories[j]):
            label_oneTweets[j] = 1
            # print('match')
            break

      # print(label_oneTweets)
      labels_all.append(label_oneTweets)
    return labels_all

  @classmethod
  def extract_one_feature(cls, dataFrame, featureName):
    '''
    Extract one feature from dataFrame
    :param dataFrame: dataFrame contains all information of all tweets
    :param featureName: name of one feature
    :return: a list of one features for all samples
    '''

    resultList = []
    for i in range(len(dataFrame)):
      resultList.append(dataFrame.iloc[i][featureName])
    return resultList

  @classmethod
  def extract_features(cls, dataFrame, featureList):
    '''
    Extract mulitple features by using extract_one_feature function
    :param dataFrame: dataFrame contains all information of all tweets
    :param featureList: a list contains names of wanted features
    :return: a dcit contains all features
    '''
    all_ft = {}
    for i in range(len(featureList)):
      all_ft[featureList[i]] = Process_labels.extract_one_feature( dataFrame, featureList[i])
    return all_ft

  @classmethod
  def extract_hashtags(cls, dataset):
    '''
    function is designed extract hashtags from dataFrame
    :param dataset: dataFrame contains all information of all tweets
    :return: a lsit contains hashtags for all samples
    '''
    dataset_after = dataset.copy()
    temp = []
    for i in range(len(dataset['entities'])):
      hashtags = ' '
      for j in range(len(dataset['entities'][i]['hashtags'])):
        hashtags += (dataset['entities'][i]['hashtags'][j]['text']) + ' '
      temp.append(hashtags)
    dataset_after['hashtags'] = temp
    return dataset_after

  @classmethod
  def extract_majority_vote_label(cls, frame):
    '''
    This function extract duplicate priority labels by majority vote method
    More specifically,if one tweet is assessed by more than two assessors,
    we use the most common label as thepriority label.
    And if one tweet is assessed by less than two assessors,
    we choose first label as the true label.
    :param frame: A dataFrame contains all labelled tweets
    :return: New dataFrame with new labels of prioritization task
    '''
    categories = ['Low', 'Medium', 'High', 'Critical']
    for idx in range(len(frame)):
      length = len(frame.loc[idx, 'priority'])
      final_label = None
      if length >= 3:
        number_count = [0, 0, 0, 0]
        for priority_label in frame.loc[idx, 'priority']:
          if priority_label == 'Low':
            number_count[0] += 1
          elif priority_label == 'Medium':
            number_count[1] += 1
          elif priority_label == 'High':
            number_count[2] += 1
          else:
            number_count[3] += 1
        final_label = categories[np.argmax(np.array(number_count))]
      else:
        final_label = frame.loc[idx, 'priority'][0]
      # print(final_label)
      frame.loc[idx, 'priority'] = final_label
    return frame

  @classmethod
  def build_single_label(cls, labels, categories):
    labels_single = {}
    for i in range(len(categories)):
      print('Currrent is processing on categorie %s'%(categories[i]))
      print()
      # if i == 3:
      #   continue
      labels_one_category = []
      for j in range(len(labels)):
        labels_one_category.append(labels[j][i])
      labels_single[i] = labels_one_category
    return labels_single

class Process_dataset():
  def generateTrainingSet(self, frame):
    '''
    This function split whole dataset into training set and
    validation set, test set and the ratio is 0.2
    :param frame: whole dataset in dataFrame
    :return: trianing set, validation set, test set
    '''
    random_frame = frame.sample(frac=1)
    train_split = int(len(random_frame) * 0.8)
    tmp_train = frame.iloc[:train_split, :]
    test_data = frame.iloc[train_split:, :]

    validation_split = int(train_split * 0.8)
    train_data = frame.iloc[:validation_split, :]
    validation_data = frame.iloc[validation_split:, :]
    return train_data, validation_data, test_data

  def generateTrainingSetCV(self, frame):
    '''
    This function split whole dataset into training set and
     test set and the ratio is 0.3
    :param frame: whole dataset in dataFrame
    :return: trianing set, test set
    '''
    random_frame = frame.sample(frac=1, random_state=1999)
    train_split = int(len(random_frame) * 0.7)
    train_data = frame.iloc[:train_split, :]
    test_data = frame.iloc[train_split:, :]
    return train_data, test_data

  def store_train_test_split_to_csv(self, df, location, target, val_ratio=0.2):
    # split dataset and store into csv file
    idx = np.arange(df.shape[0])

    np.random.shuffle(idx)

    val_size = int(len(idx) * val_ratio)
    if not os.path.exists('cache/' + location):
      os.makedirs('cache/' + location)

    df[['full_text', 'entities', 'favorite_count', target]].to_csv('cache/' + location + '/dataset_all.csv',
                                                                   index=False)

    # df_tmp = df.iloc[idx[val_size:]
    # df_test = df.iloc[idx[:val_size]
    df.iloc[idx[val_size:], :][['full_text', 'entities', 'favorite_count', target]].to_csv(
      'cache/' + location + '/dataset_test.csv',
      index=False)

    df.iloc[idx[val_size:], :][['full_text', 'entities', 'favorite_count', target]].to_csv(
      'cache/' + location + '/dataset_train.csv', index=False)

    # df.iloc[idx[:val_size], :][['full_text','entities','favorite_count', target]].to_csv(
    #     'cache/dataset_val.csv', index=False
    # )


class Classical_classifiers():

  def build_best_lr_priority(self, cf, vect_count, category, increaseTime, trainF, trainL, testF, testL):
    '''
    This function build best logistic regression model for priorization task
    :param cf: classifier to use
    :param vect_count: Count Vecoterizer
    :param category: indicate current processing category
    :param increaseTime: Over-sample factor
    :param trainF:  Training features
    :param trainL: Training labels
    :param testF: Testing features
    :param testL: Testing labels
    :return: trained classifier, pipeline, featureUnion function,
              featureUnion array of test features, index of all error, dataFrame of all errors
    '''
    # Step 1: build the up-sampling set based on the best over-sample factor
    # protect the original dataset
    start_time = time.time()
    trainSet_temp = copy.deepcopy(trainF)
    label_temp = copy.deepcopy(trainL)

    # Find all sample belong to current processing category
    idxs = np.where(np.array(label_temp) == category)[0]

    # iterate all sample belong to current processing category and duplicate all features
    for idx in list(idxs):
      # Duplicate (increaseTime) times
      for j in range(increaseTime):
        trainSet_temp['hashtags'].append(trainSet_temp['hashtags'][idx])
        trainSet_temp['full_text'].append(trainSet_temp['full_text'][idx])
        trainSet_temp['favorite_count'].append(trainSet_temp['favorite_count'][idx])
        label_temp.append(category)
    # Rename  varibles
    trainF_up = trainSet_temp
    trainL_up = label_temp

    # Step 2: featureunion the up-sample data set
    fu_best = FeatureUnion(
      [('full_text', myPipeline([('selector', ItemSelector(key='full_text')), ('cv', vect_count),
                                 ])),
       ('hashtags', myPipeline([('selector', ItemSelector(key='hashtags')),
                                ('cv', vect_count),
                                ])),
       # ('favorite_count',myPipeline([('selector', numericalTransformer(key='favorite_count')),
       # ])),
       ], n_jobs=-1)

    # Use featureUnion to fit and transfrom training set and test set
    fu_best.fit(trainF_up)
    trainF_up_union = fu_best.transform(trainF_up)
    trainF_union = fu_best.transform(trainF)
    testF_union = fu_best.transform(testF)

    # Build pipeline for featureUnion and classifier
    pipe = Pipeline([('fu', fu_best), ('lr', cf)

                     ])
    # print('shape')
    # print(trainF_up.shape)
    # print(len(trainL_up))
    pipe.fit(trainF_up, trainL_up)

    # Step3: train the classifiers and evaluate the result
    # Best LR without up-sampling
    print("Display result without up-sampleing: ")
    runningSet = [trainF_union, trainL, testF_union, testL]
    Classical_classifiers.lr_priority(runningSet, cf)

    # Best LR with up-sampling
    print("Display result with up-sampleing: ")
    # model_name = "The best LR result with featureUnion"
    runningSet = [trainF_up_union, trainL_up, testF_union, testL]
    Classical_classifiers.lr_priority(runningSet, cf)
    # evaluate_one_lr(cf, runningSet, category)

    # Evaluate the testset
    predictions = cf.predict(testF_union)
    # display some metrics
    print(classification_report(predictions, testL, digits=4))
    # Get index of error and store detial of errors in a dataFrame
    errors_idxs = Evaluate_model.get_errors_indexs(testL, predictions, testF)
    all_errors = Evaluate_model.get_errors(testL, predictions, testF)
    end_time = time.time()
    eclipsed_time = Scoring.format_time(end_time - start_time)
    print("This compuation takes:", eclipsed_time)

    return cf, pipe, fu_best, testF_union, errors_idxs, all_errors

  def build_default_lr_priority(self, location, category, increaseTime, trainF, trainL, testF, testL, tokenizer,
                                count_vect):
    '''
     This function build  logistic regression model with default setting for priorization task
     :param location: indicate location of current data set
     :param category: indicate current processing category
     :param increaseTime: Over-sample factor
     :param trainF:  Training features
     :param trainL: Training labels
     :param testF: Testing features
     :param testL: Testing labels
     :param tokenizer: tokenizer to tokenize

     :return: None
     '''
    # Build Count Vectorizer and default logistic regression model
    start_time = time.time()
    vect_count = CountVectorizer(tokenizer=tokenizer, )
    cf = LogisticRegression(max_iter=300)

    # Step 1: build the up-sampling set based on the best increaseTime
    trainSet_temp = copy.deepcopy(trainF)
    label_temp = copy.deepcopy(trainL)

    # Find all sample belong to current processing category
    idxs = np.where(np.array(label_temp) == category)[0]

    # iterate all sample belong to current processing category and duplicate all features
    for idx in list(idxs):
      # Duplicate (increaseTime) times
      for j in range(increaseTime):
        trainSet_temp['hashtags'].append(trainSet_temp['hashtags'][idx])
        trainSet_temp['full_text'].append(trainSet_temp['full_text'][idx])
        trainSet_temp['favorite_count'].append(trainSet_temp['favorite_count'][idx])
        label_temp.append(category)
    # Rename  varibles
    trainF_up = trainSet_temp
    trainL_up = label_temp

    # Step 2: featureunion the up-sample data set
    fu_best = FeatureUnion([('full_text', myPipeline([('selector', ItemSelector(key='full_text')), ('cv', count_vect),
                                                      ])),
                            ('hashtags', myPipeline([('selector', ItemSelector(key='hashtags')),
                                                     ('cv', vect_count),
                                                     ])),
                            # ('favorite_count',myPipeline([('selector', numericalTransformer(key='favorite_count')),
                            # ])),
                            ], n_jobs=-1)

    # Use featureUnion to fit and transfrom training set and test set
    fu_best.fit(trainF_up)
    trainF_up_union = fu_best.transform(trainF_up)
    fu_best.fit(trainF)
    trainF_union = fu_best.transform(trainF)
    testF_union = fu_best.transform(testF)
    # Build pipeline for featureUnion and classifier
    pipe = Pipeline([('fu', fu_best), ('lr', cf)
                     ])
    pipe.fit(trainF_up, trainL_up)

    # Step3: train the classifiers and evaluate the result
    # Default LR without up-sampling
    print("Display result without up-sampleing: ")
    runningSet = [trainF_union, trainL, testF_union, testL]
    Classical_classifiers.lr_priority(runningSet)
    # display_result_without_upsample(cf, category, runningSet)

    print("Display result with up-sampleing: ")
    # model_name = "The best LR result with featureUnion"
    # testF_union = more[2]
    # testL = more[3]
    runningSet = [trainF_up_union, trainL_up, testF_union, testL]
    Classical_classifiers.lr_priority(runningSet)
    # evaluate_one_lr(cf, runningSet, category)

    # cf.fit()
    # predictions = cf.predict(testF_union)

    # print(classification_report(predictions, testL, digits=4))
    # errors_idxs = get_errors_indexs(testL, predictions, testF)
    # all_errors = get_errors(testL, predictions, testF)

    end_time = time.time()
    eclipsed_time = Scoring.format_time(end_time - start_time)
    print("This compuation takes:", eclipsed_time)
    return  # cf, pipe, fu_best,  errors_idxs, all_errors

  def build_best_lr(self, location, cf, vect_count, vect_tfidf, category, increaseTime,
                    trainF, trainL_single, testF, testL_single):
    '''
    This function build the best logistic regression model based
    on the best hyper-parameters and evaluate it
    :param location: indicate location of current data set
    :param cf: pass the classifiers to this function
    :param vect_count: Count Vectorizer
    :param vect_tfidf: TF-IDF Vectorizer
    :param category:  indicate current processing category
    :param increaseTime: Over-sample factor
    :param trainF:  Train features
    :param trainL: Train labels
    :param testF: Test features
    :param testL_single:  A dict testing labels for every categories
    :return: trained classifier, pipeline , featureUnion function,
            featureUnion array of test features, index of all error, dataFrame of all errors
    '''
    # read over-sample dataset from csv file
    start_time = time.time()
    df_tmp = pd.read_csv('cache/' + location + '/upsampleData_' + str(category) + '_' + str(increaseTime) + '.csv')
    # Load features and labels from dataFrame
    trainF_up_one = {'full_text': df_tmp['full_text'], 'hashtags': df_tmp['hashtags'],
                     'favorite_count': df_tmp['favorite_count']}
    trainL_up_one = list(df_tmp['target'])

    # featureunion the up-sample data set
    fu_best = FeatureUnion(
      [('full_text', myPipeline([('selector', ItemSelector(key='full_text')), ('tf_idf', vect_tfidf),
                                 ])),
       ('hashtags', myPipeline([('selector', ItemSelector(key='hashtags')),
                                ('cv', vect_count),
                                ])),
       # ('favorite_count',myPipeline([('selector', numericalTransformer(key='favorite_count')),
       # ])),
       ], n_jobs=-1)
    # Use featureUnion to fit and transfrom training set and test set
    fu_best.fit(trainF_up_one)
    trainF_up_one_union = fu_best.transform(trainF_up_one)
    fu_best.fit(trainF)
    trainF_union = fu_best.transform(trainF)
    testF_union = fu_best.transform(testF)

    # Build pipeline for featureUnion and classifier
    pipe = Pipeline([('fu', fu_best), ('lr', cf)
                     ])
    pipe.fit(trainF_up_one, trainL_up_one)

    print("Display result without up-sampleing: ")
    runningSet = [trainF_union, trainL_single[category], testF_union, testL_single[category]]
    Evaluate_model.display_result_without_upsample(cf, category, runningSet)

    print("Display result with up-sampleing: ")
    model_name = "The best LR result with featureUnion"
    # print(trainF_up_one_union.shape)
    # print(len(trainL_up_one))

    # train the classifiers and evaluate the result
    runningSet = [trainF_up_one_union, trainL_up_one, testF_union, testL_single[category]]
    Evaluate_model.evaluate_one_cf(cf, runningSet, category)

    # Get index of error and store detial of errors in a dataFrame
    predictions = cf.predict(testF_union)
    errors_idxs = Evaluate_model.get_errors_indexs(testL_single[category], predictions, testF)
    all_errors = Evaluate_model.get_errors(testL_single[category], predictions, testF)

    end_time = time.time()
    eclipsed_time = Scoring.format_time(end_time - start_time)
    print("This compuation takes:", eclipsed_time)
    return cf, pipe, fu_best, testF_union, errors_idxs, all_errors

  def build_best_lr_without_featureunion(self, location, cf, vect_tfidf, category, increaseTime,
                                         trainF, trainL_single, testF, testL_single):
    '''
    This function build logistic regression model through only use full-text feature
    :param location: indicate location of current data set
    :param cf: pass the classifiers to this function
    :param vect_count: Count Vectorizer
    :param vect_tfidf: TF-IDF Vectorizer
    :param category:  indicate current processing category
    :param increaseTime: Over-sample factor
    :param trainF:  Training features
    :param trainL: Training labels
    :param testF: Testing features
    :param testL_single:  A dict testing labels for every categories
    :return: trained classifier, tranformed test featrue array
    '''
    start_time = time.time()

    df_tmp = pd.read_csv('cache/' + location + '/upsampleData_' + str(category) + '_' + str(increaseTime) + '.csv')
    trainF_up_one = df_tmp['full_text']
    trainL_up_one = list(df_tmp['target'])

    trainF_up_one = vect_tfidf.transform(trainF_up_one)
    testF_transformed = vect_tfidf.transform(testF['full_text'])

    cf.fit(trainF_up_one, trainL_up_one)

    end_time = time.time()
    eclipsed_time = Scoring.format_time(end_time - start_time)
    print("This compuation takes:", eclipsed_time)

    return cf, testF_transformed

  def build_default_lr(self, location, category, increaseTime, trainF, trainL_single,
                       testF, testL_single, tokenizer):
    '''

    :param location: indicate location of current data set
    :param category: indicate current processing category
    :param increaseTime: Over-sample factor
    :param trainF: Training features
    :param trainL_single: A dict of training labels for every categories
    :param testF: Test features
    :param testL_single: A dict of testing labels for every categories
    :param tokenizer: tokenizer used to tokenize
    :return: trained classifier, pipeline , featureUnion array of test features,
              index of all error, dataFrame of all errors
    '''
    # Build vectorizers and classifier with default setting
    start_time = time.time()

    vect_count = CountVectorizer(tokenizer=tokenizer, )
    vect_tfidf = TfidfVectorizer(tokenizer=tokenizer, )
    cf = LogisticRegression()

    # Read over-sample data set from csv file
    df_tmp = pd.read_csv('cache/' + location + '/upsampleData_' + str(category) + '_' + str(increaseTime) + '.csv')
    trainF_up_one = {'full_text': df_tmp['full_text'], 'hashtags': df_tmp['hashtags'],
                     'favorite_count': df_tmp['favorite_count']}
    trainL_up_one = list(df_tmp['target'])
    # featureunion the up-sample data set
    fu_best = FeatureUnion(
      [('full_text', myPipeline([('selector', ItemSelector(key='full_text')), ('tf_idf', vect_tfidf),
                                 ])),
       ('hashtags', myPipeline([('selector', ItemSelector(key='hashtags')),
                                ('cv', vect_count),
                                ])),
       # ('favorite_count',myPipeline([('selector', numericalTransformer(key='favorite_count')),
       # ])),
       ], n_jobs=-1)

    # Use featureUnion to fit and transfrom training set and test set
    fu_best.fit(trainF_up_one)
    trainF_up_one_union = fu_best.transform(trainF_up_one)
    trainF_union = fu_best.transform(trainF)
    testF_union = fu_best.transform(testF)

    # Build pipeline for featureUnion and classifier
    pipe = Pipeline([('fu', fu_best), ('lr', cf)
                     ])
    pipe.fit(trainF_up_one, trainL_up_one)

    print("Display result without up-sampleing: ")
    runningSet = [trainF_union, trainL_single[category], testF_union, testL_single[category]]
    Evaluate_model.display_result_without_upsample(cf, category, runningSet)

    print("Display result with up-sampleing: ")
    model_name = "The best LR result with featureUnion"
    # print(trainF_up_one_union.shape)
    # print(len(trainL_up_one))

    # train the classifiers and evaluate the result
    runningSet = [trainF_up_one_union, trainL_up_one, testF_union, testL_single]
    Evaluate_model.evaluate_one_cf(cf, runningSet, category)

    # Get index of error and store detial of errors in a dataFrame
    predictions = cf.predict(testF_union)
    errors_idxs = Evaluate_model.get_errors_indexs(testL_single[category], predictions, testF)
    all_errors = Evaluate_model.get_errors(testL_single[category], predictions, testF)

    end_time = time.time()
    eclipsed_time = Scoring.format_time(end_time - start_time)
    print("This compuation takes:", eclipsed_time)

    return cf, pipe, fu_best, errors_idxs, all_errors

  def DummyResult(self, runningSet):
    '''
    This function use dummy classifier to evaluate
    :param runningSet: A list contrains training features, training labels
                        test features, test labels
    :return: None
    '''
    # unpack running set
    start_time = time.time()
    train_features, train_labels = runningSet[:2]

    from sklearn.dummy import DummyClassifier
    print("Evaluate dummy classifier with strategy of most_frequent")
    dummy_cf = DummyClassifier(strategy='most_frequent')
    dummy_cf.fit(train_features, train_labels)
    cv_scores = Evaluate_model.total_evaluation(dummy_cf, 'Dummy', runningSet, multi_sign=True)
    # record_result('dummy_most_train',precision_train,recall_train,f1_train)
    # record_result('dummy_most_test',precision_test,recall_test,f1_test)

    # DummyClassifier with stratified

    from sklearn.dummy import DummyClassifier
    print("Evaluate dummy classifier with strategy of stratified")
    dummy_cf = DummyClassifier(strategy='stratified')
    dummy_cf.fit(train_features, train_labels)
    cv_scores = Evaluate_model.total_evaluation(dummy_cf, 'Dummy', runningSet, multi_sign=True)
    # record_result('dummy_stratified_train',precision_train,recall_train,f1_train)
    # record_result('dummy_stratified_test',precision_test,recall_test,f1_test)
    end_time = time.time()
    eclipsed_time = Scoring.format_time(end_time - start_time)
    print("This compuation takes:", eclipsed_time)


  def lr_Binary(self, runningSet, category, categories):
    '''
    This function build and evaluate a binary logistic regression function
    :param runningSet: A list contrains training features, training labels
                          test features, test labels
    :param category: current processing category
    :param categories: names of all categories
    :return: None
    '''
    start_time = time.time()
    from sklearn.linear_model import LogisticRegression
    # unpack
    train_set = runningSet[0]
    train_labels = runningSet[1]
    testL_union_tmp = runningSet[2]
    testL = runningSet[3]

    # LogisticRegression

    lr = LogisticRegression(max_iter=300)
    # lr.fit(train_set,train_labels)
    cv_scores = Evaluate_model.total_evaluation(lr, 'LogisticRegression', runningSet)
    # evaluate on the test set
    lr.fit(train_set, train_labels)
    predictions = lr.predict(testL_union_tmp)
    precision = precision_score(predictions, testL[category])
    recall = recall_score(predictions, testL[category])
    accuracy = accuracy_score(predictions, testL[category])
    f1 = fbeta_score(predictions, testL[category], 1, )

    # store test result
    test_scores = {'accuracy': accuracy, 'precision': precision, 'f1': f1, 'recall': recall}
    print("Test result:", test_scores)
    # record_nine_result(cv_scores, test_scores, categories[category])

    end_time = time.time()
    eclipsed_time = Scoring.format_time(end_time - start_time)
    print("This compuation takes:", eclipsed_time)

  def lr_priority(self, runningSet, print_out_detail=False, cf=None):
    '''
    This function build and evaluate a  logistic regression function
    for prioritization task
    :param runningSet: A list contrains training features, training labels
                          test features, test labels
    :param print_out_detail: current processing category
    :param cf: classifier to use
    :return: None
    '''
    start_time = time.time()

    train_set = runningSet[0]
    train_labels = runningSet[1]
    testF = runningSet[2]
    testL = runningSet[3]
    if cf == None:
      lr = LogisticRegression()
      # max_iter=300
    else:
      lr = cf
    # lr.fit(train_set,train_labels)
    cv_scores = Evaluate_model.total_evaluation(lr, 'LogisticRegression', runningSet, print_out_detail)

    lr.fit(train_set, train_labels)
    # choose metrics based on different number of categories
    type_number = len(set(testL))
    if type_number <= 2:
      predictions = lr.predict(testF)
      precision = precision_score(predictions, testL)
      recall = recall_score(predictions, testL)
      accuracy = accuracy_score(predictions, testL)
      f1 = fbeta_score(predictions, testL, 1, )
    else:
      predictions = lr.predict(testF)
      precision = precision_score(predictions, testL, average='macro')
      recall = recall_score(predictions, testL, average='macro')
      accuracy = accuracy_score(predictions, testL)
      f1 = fbeta_score(predictions, testL, 1, average='macro')

    test_scores = {'accuracy': accuracy, 'precision': precision, 'f1': f1, 'recall': recall}
    print("Test result:", test_scores)
    print(classification_report(predictions, testL, digits=4))

    end_time = time.time()
    eclipsed_time = Scoring.format_time(end_time - start_time)
    print("This compuation takes:", eclipsed_time)

    return cv_scores, test_scores

  @classmethod
  def svm(cls, runningSet, vectorizers,  section_category, over_sample_targets, increase_factor, print_out_detail=False):
    '''
    Build a support vector machine to evaluate given dataset
    :param runningSet: A list contrains training features, training labels
                        test features, test labels
    :param print_out_detail: a sign whether print out verbose result
    :return:
    '''
    start_time = time.time()
    from sklearn.svm import SVC
    # unpack
    train_features, train_labels, test_features,  test_labels   = runningSet[0], runningSet[1], runningSet[2], runningSet[3]

    # Step 1: Over-sample data set
    print("Over-sample data set")
    train_fea_over_sample, train_labels_over_sample = Up_sample.over_sample_from_frame(train_features, train_labels, categories= over_sample_targets, increase_factor=increase_factor)

    # Step 2: featureunion the up-sample data set
    print("feature-union")
    fu = Feature_union_pipeline.build_fu_from_feature_list(vectorizers[0], vectorizers[1])

    # Use featureUnion to fit and transform training set and test set
    fu.fit(train_fea_over_sample)
    train_fea_over_sample_union = fu.transform(train_fea_over_sample)
    train_fea_union = fu.transform(train_features)
    test_fea_union = fu.transform(test_features)


    # Step3: train the classifiers and evaluate the result



    #  1. feature union
    runningSet = [train_fea_union, train_labels, test_fea_union, test_labels]
    print("Train with feature union")
    # build the classiifer
    svm_cf = SVC()
    cv_scores = Evaluate_model.total_evaluation(svm_cf, 'SVM', runningSet, print_out_detail)

    #  2. over-sample + feature union
    print("Train with feature union + over-sample")
    runningSet = [train_fea_over_sample_union, train_labels_over_sample, test_fea_union,test_labels]
    svm_cf = SVC()
    cv_scores = Evaluate_model.total_evaluation(svm_cf, 'SVM', runningSet, print_out_detail)

    end_time = time.time()
    eclipsed_time = Scoring.format_time(end_time - start_time)
    print("This compuation takes:", eclipsed_time)
    return end_time - start_time

  @classmethod
  def DecisionTreeCf(cls, runningSet, vectorizers,  section_category, over_sample_targets, increase_factor, print_out_detail=False):
    '''
    This function use decision tree to evaluate given dataset
    :param runningSet: A list contrains training features, training labels
                        test features, test labels
    :return: None
    '''
    # unpack running set
    start_time = time.time()

    train_features, train_labels, test_features, test_labels = runningSet[0], runningSet[1], runningSet[2], runningSet[3]

    # Step 1: Over-sample data set
    train_fea_over_sample, train_labels_over_sample = Up_sample.over_sample_from_frame(train_features, train_labels, categories= over_sample_targets, increase_factor=increase_factor)

    # Step 2: featureunion the up-sample data set
    fu = Feature_union_pipeline.build_fu_from_feature_list(vectorizers[0], vectorizers[1])

    # Use featureUnion to fit and transfrom training set and test set
    fu.fit(train_fea_over_sample)
    train_fea_over_sample_union = fu.transform(train_fea_over_sample)
    train_fea_union = fu.transform(train_features)
    test_fea_union = fu.transform(test_features)

    # Step3: train the classifiers and evaluate the result
    #  1. feature union
    runningSet = [train_fea_union, train_labels, test_fea_union, test_labels]
    print("Train with feature union")
    # build the classiifer
    decisionTreeCf = DecisionTreeClassifier()
    cv_scores = Evaluate_model.total_evaluation(decisionTreeCf, 'DT', runningSet, print_out_detail)

    #  2. over-sample + feature union
    print("Train with feature union + over-sample")
    runningSet = [train_fea_over_sample_union, train_labels_over_sample, test_fea_union,test_labels]
    decisionTreeCf = DecisionTreeClassifier()
    cv_scores = Evaluate_model.total_evaluation(decisionTreeCf, 'DT', runningSet, print_out_detail)

    # from sklearn.tree import plot_tree
    # plt.figure()
    # plot_tree(decisionTreeCf)
    # plt.savefig('./DecisionTrees.png', dpi=400, )
    # evaluate on the test set


    end_time = time.time()
    eclipsed_time = Scoring.format_time(end_time - start_time)
    print("This compuation takes:", eclipsed_time)
    return end_time - start_time

  @classmethod
  def cross_event_validate_lr_info_task(cls, section_category, a_batch, b_batch, increase_factor,
                                        full_text_vectorizer, hash_tags_vectorizer
                        ):
    start_time = time.time()
    import json
    # training
    a_all_data, a_labels  = a_batch[0], a_batch[1][section_category]
    b_all_data, b_labels = b_batch[0], b_batch[1][section_category]

    print('Current is process category %d' %(section_category))
    #
    featureList = ['full_text','hashtags','favorite_count'] #,'favorited'
    a_extracted = Process_labels.extract_features(a_all_data,featureList)
    b_extracted = Process_labels.extract_features(b_all_data,featureList)

    # Over-sample both data set
    a_extracted, a_labels = Up_sample.over_sample_from_frame(a_extracted, a_labels, categories= [1], increase_factor= increase_factor)
    b_extracted, b_labels = Up_sample.over_sample_from_frame(b_extracted, b_labels, categories= [1], increase_factor= increase_factor)

    # Train on dataset-a and test on dataset-b
    print('Cross event validation')
    fu = Feature_union_pipeline.build_fu_from_feature_list(full_text_vectorizer, hash_tags_vectorizer)
    fu.fit(a_extracted)
    a_features_union = fu.transform(a_extracted)
    b_features_union = fu.transform(b_extracted)

    lr = LogisticRegression(max_iter=300)
    lr.fit(a_features_union, a_labels)
    print('Training result:')

    scores = Evaluate_model.cross_validation_result(LogisticRegression(max_iter=300), a_features_union, a_labels)


    predictions = lr.predict(b_features_union)
    print("Test result on a:")
    print(classification_report(predictions,  b_labels, digits=3))


    # Train on dataset-b and test on dataset-a
    fu = Feature_union_pipeline.build_fu_from_feature_list(full_text_vectorizer, hash_tags_vectorizer)
    fu.fit(b_extracted)
    a_features_union = fu.transform(a_extracted)
    b_features_union = fu.transform(b_extracted)

    scores = Evaluate_model.cross_validation_result(LogisticRegression(max_iter=300), b_features_union, b_labels)

    lr = LogisticRegression(max_iter=300)
    lr.fit(b_features_union, b_labels)
    predictions = lr.predict(a_features_union)

    print("Test result on b:")
    print(classification_report(predictions,  a_labels, digits=3))

    end_time = time.time()
    eclipsed_time = Scoring.format_time(end_time - start_time)
    print("This compuation takes:", eclipsed_time)


class Plot_figures():
  def plot_metrics_Upsample(self, records, category, phase):
    '''
    This function plot figures of over-sample result
    :param records: record dict of all metircs
    :param category: current processing category
    :param phase: text, indicates the phase of training
    :return:
    '''
    plt.figure()
    plt.title('%s result for %s .' % (phase, category))
    length = len(records[category]['accuracy'])
    plt.plot(range(length), records[category]['accuracy'], label='accuracy')
    plt.plot(range(length), records[category]['f1'], label='f1')
    plt.plot(range(length), records[category]['recall'], label='recall')
    plt.plot(range(length), records[category]['precision'], label='precision')
    plt.xlabel('Increase Times')
    plt.ylabel('Values')
    # xticks = np.arange(16)
    # plt.xticks(xticks)
    plt.legend()
    plt.savefig(str(category) + '_' + phase + '.png', dpi=300, bbox_inches='tight')
    plt.show()

  def plot_pie_priority_count(self, frame, location):
    '''
    This function plot pie figure for priority labels
    :param frame:  whole dataset in dataFrame
    :param location: current processing location
    :return: None
    '''
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    data = [x for x in frame.priority.value_counts()]
    ingredients = [x for x in frame.priority.value_counts().index]

  def plotCM(self, title, classifier, X_test, y_test):
    '''
    This function plot confusion matrix of given classifier
    :param title: text title for figure
    :param classifier: classifier to evaluate
    :param X_test: trianing features
    :param y_test: true labels
    :return:
    '''
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 xticks_rotation='vertical',
                                 cmap=plt.cm.Blues, ax=ax,
                                 normalize='true')

    disp.ax_.set_title(title)
    plt.savefig(title + '_' + 'CM' + '_' + 'priority' + '.png', dpi=200, bbox_inches='tight')

  def types_statistics(self, frame, location):
    '''
    Print out some statistics of given location
    :param frame: whoel data set
    :param location: name of processing location
    :return:
    '''
    # detail of catgories information
    print("Showing statistics for types classification: ")
    print(location + ":")
    print(frame.categories.value_counts())
    Plot_figures.plot_pie_priority_count(frame, location)

    # Figure of favorite count
    plt.figure()
    frame.favorite_count.value_counts().nlargest(20).plot(kind='bar')
    plt.title('The number of favorite count for ' + location)
    plt.xlabel('The times of favorite')
    plt.ylabel('The number of tweets')
    if not os.path.exists('cache/' + location + '/figures/'):
      os.makedirs('cache/' + location + '/figures/')
    plt.savefig('cache/' + location + '/figures/' + location + '_favorite_count.png', ppi=400, bbox_inches='tight')
    plt.show()

    # Figure of count of  priority label
    data = frame
    categories = list(set(data.priority))
    sns.set(font_scale=2)
    plt.figure()
    ax = sns.barplot(categories, data.priority.value_counts())
    plt.title("Priority in each category")
    plt.ylabel("Number count")
    plt.xlabel("Priority type")
    plt.savefig('cache/' + location + '/figures/' + location + '_priority_count.png', ppi=400, bbox_inches='tight')
    plt.show()

    # Figure of number of label of each tweet
    sns.set(font_scale=2)
    plt.figure()
    index_len = {}
    for index in data.categories.value_counts().index:
      if len(index) in index_len.keys():
        index_len[len(index)] += 1
      else:
        index_len[len(index)] = 1
    index_len_keys = list(index_len.keys())
    index_len_values = list(index_len.values())
    ax = sns.barplot(index_len_keys, index_len_values)
    plt.title("Labels number")
    plt.ylabel("Number count")
    plt.xlabel("Number of labels")
    plt.savefig('cache/' + location + '/figures/' + location + '_label_number_count.png', ppi=400, bbox_inches='tight')
    plt.show()

    # WorldCloud
    from wordcloud import WordCloud, STOPWORDS
    plt.figure()
    text = data.categories.values
    text_all = []
    for i in range(len(text)):
      for j in range(len(text[i])):
        text_all.append(text[i][j])

    cloud_toxic = WordCloud(stopwords=STOPWORDS,
                            background_color='black',
                            collocations=False,
                            ).generate(" ".join(text_all))
    plt.axis('off')
    plt.title("Catergories", fontsize=30)
    plt.savefig('cache/' + location + '/figures/' + location + '_word_cloud.png', ppi=400, bbox_inches='tight')
    plt.imshow(cloud_toxic)

    #
    print("All labeled types: ", set(text_all))




class numericalTransformer_gridSearch(BaseEstimator, TransformerMixin):
  """For data grouped by feature, select subset of data at a provided key.    """

  def __init__(self, key):
    self.key = key

  def fit(self, x, y=None):
    return self

  def transform(self, data_dict):
    tmp = []
    for i in range(len(data_dict)):
      tmp.append(data_dict[i][self.key])

    return np.array(tmp, dtype='float32').reshape(-1, 1)


class myPipeline(Pipeline):
  def get_feature_names(self):
    for name, step in self.steps:
      if isinstance(step, TfidfVectorizer):
        print('TfidfVectorizer')
        return step.get_feature_names()
      elif isinstance(step, CountVectorizer):
        print('CountVectorizer')
        return step.get_feature_names()
      # elif isinstance(step,numericalTransformer):
      #     print('Numerical')
      #     return str(set(trainF_up_one['favorite_count']))



class ItemSelector_gridSearch(BaseEstimator, TransformerMixin):
  """For data grouped by feature, select subset of data at a provided key.    """

  def __init__(self, key):
    self.key = key

  def fit(self, x, y=None):
    return self

  def transform(self, data_dict):
    tmp = []

    for i in range(len(data_dict)):
      tmp.append(data_dict[i][self.key])

    return np.array(tmp)

class ItemSelector(BaseEstimator, TransformerMixin):
  """For data grouped by feature, select subset of data at a provided key.    """

  def __init__(self, key):
    self.key = key

  def fit(self, x, y=None):
    return self

  def transform(self, data_dict):
    return data_dict[self.key]


class numericalTransformer(BaseEstimator, TransformerMixin):
  """For data grouped by feature, select subset of data at a provided key.    """

  def __init__(self, key):
    self.key = key

  def fit(self, x, y=None):
    return self

  def transform(self, data_dict):
    return np.array(data_dict[self.key], dtype='float32').reshape(-1, 1)

class w2vTransformer(TransformerMixin):
    """
    Wrapper class for running word2vec into pipelines and FeatureUnions
    """

    def __init__(self, word2vec, **kwargs):
      self.word2vec = word2vec
      self.kwargs = kwargs
      self.dim = len(word2vec.values())

    def fit(self, x, y=None):
      return self

    def transform(self, X):
      return np.array([
        np.mean([self.word2vec.wv[w] for w in words if w in self.word2vec]
                or [np.zeros(self.dim)], axis=0)
        for words in X
      ])
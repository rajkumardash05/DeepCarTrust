# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 10:45:17 2020

@author: rajku
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import pdb
import Deep_Models as DM
import Model_Evaluation as ME
from sklearn.metrics import accuracy_score

tf.logging.set_verbosity(tf.logging.ERROR)


filename = "D:\\Python Code Repo\\Dataset\\Autonomous Vehicle\\Data_ML\\Data_Combined_Manually.xlsx"
filename_pred = "D:\\Python Code Repo\\Dataset\\Autonomous Vehicle\\Data_ML\\Predictions.xlsx"
filename_pred_CF = "D:\\Python Code Repo\\Dataset\\Autonomous Vehicle\\Data_ML\\Predictions_CF.xlsx"
filename_reg = "D:\\Python Code Repo\\Dataset\\Autonomous Vehicle\\Data_ML\\Predictions_Regression.xlsx"
filename_MOC = "D:\\Python Code Repo\\Dataset\\Autonomous Vehicle\\Data_ML\\Predictions_Multi_Class.xlsx"
filename_BC = "D:\\Python Code Repo\\Dataset\\Autonomous Vehicle\\Data_ML\\Predictions_Binary_Class.xlsx"
metrics_file = "D:\\Python Code Repo\\Dataset\\Autonomous Vehicle\\Data_ML\\Metrics_File.txt"
filename_car = "D:\\Python Code Repo\\Dataset\\Autonomous Vehicle\\Data_ML\\Data_Set_Car.xlsx"
filename_car_processed = "D:\\Python Code Repo\\Dataset\\Autonomous Vehicle\\Data_ML\\Data_Set_Car_Processed.xlsx"

filename_MOC_car = "D:\\Python Code Repo\\Dataset\\Autonomous Vehicle\\Data_ML\\Predictions_Multi_Class_Car.xlsx"
metrics_file_car = "D:\\Python Code Repo\\Dataset\\Autonomous Vehicle\\Data_ML\\Metrics_File_Car.txt"
filename_BC_car = "D:\\Python Code Repo\\Dataset\\Autonomous Vehicle\\Data_ML\\Predictions_Binary_Class_Car.xlsx"


filename_analysis = "D:/Python Code Repo/Dataset/Autonomous Vehicle/Data_ML/Final Results/Predictions_Multi_Class_Car_PR_ROC.xlsx"
filename_analysis_DNN = "D:/Python Code Repo/Dataset/Autonomous Vehicle/Data_ML/Final Results/Predictions_PR_ROC.xlsx"
filename_analysis_Keras = "D:/Python Code Repo/Dataset/Autonomous Vehicle/Data_ML/Final Results/Predictions_PR_ROC_Keras.xlsx"


features = ['Flowrate', 'Speed', 'Phasetime']
data_car_columns = ['Vehicle no', 'Time 1', 'Time 2 (T1+30 sec)', 'Speed 1', 'Speed 2', 'Distance 1', 'Distance 2', 'Camera', 'Lidar', 'Radar']
target_car = ['C', 'L', 'R']
target =   ['Two_Class']
target_CF =   ['Class_Faulty']
target_MC = ['Class_Label']
target_Reg = ['Trustworthiness']
target_MO = ['Camera_Faulty',	'Lidar_Faulty',	'Radar_Faulty',	'Acoustic_Faulty',	'GPS_Faulty',	'Message_Faulty']
sheet_name = ['Trustworthy_Data', 'Untrustworthy_Data', 'Untrustworthy_MO']
train_test_pred_fraction = [0.6, 0.2, 0.2]
hidden_units_spec = [1024, 512, 256]#, 20, 10]
dropout_spec = [0.5, 0.4, 0.3]#, 0.05, 0.05]
dropout = 0.2
epochs_spec = 15
steps_spec = 2000
output_files = [filename_pred, filename_reg, filename_pred_CF]
batch_norm = False
n_classes = 2
n_classes_MC = 6

#batch_normed = tf.keras.layers.BatchNormalization()(hidden, training=in_training_mode)


def read_data(filename, features, target, sheet_name):
    df = pd.read_excel(filename, sheet_name = sheet_name)
    
    df = df[features + target]
    
    return df

def read_sample_weight(filename, sample_weight, sheet_name):
    df = pd.read_excel(filename, sheet_name = sheet_name)
    
    df = df[sample_weight]
    
    return df

def prepare_data(df1, df2, Shuffle = True):
    
    df = df1.append(df2).reset_index(drop = True)
    
    if Shuffle == True:
        df = df.reindex(np.random.permutation(df.index))  # need to shuffle the data to get a mix of both classes    
    
    return df

def read_data_car(filename_car, data_car_columns, sheet_name = 'data_car'):
    # Reading the data related to individual cars' speed and distance.
    df = pd.read_excel(filename_car, sheet_name = sheet_name)
    df = df[data_car_columns]
    
    columns = ['A1', 'D1', 'SD1', 'A2', 'D2', 'SD2', 'A3', 'RS3', 'D3', 'SD3', 'A4', 'RS4', 'D4', 'SD4',
               'A5', 'RS5', 'D5', 'SD5', 'A6', 'RS6', 'D6', 'SD6', 'A7', 'RS7', 'D7', 'SD7', 'A8', 'RS8', 'D8', 'SD8',
               'C', 'L', 'R']
    data = pd.DataFrame([], columns = columns)
    
    
    frame = []
    for x in list(df.index):
        if df.iloc[x, :]['Vehicle no'] == 9:
            data = data.append(pd.DataFrame([frame], columns = columns), ignore_index = True)
            frame = []
            continue
        frame.append((df.iloc[x, :]['Speed 1'] - df.iloc[x, :]['Speed 2'])/0.5)
        
        if df.iloc[x, :]['Vehicle no'] != 1 and df.iloc[x, :]['Vehicle no'] != 2:
#            pdb.set_trace()
            rs1 = (df.iloc[x, :]['Speed 1'] - df.iloc[x-2, :]['Speed 1'])
            rs2 = (df.iloc[x, :]['Speed 2'] - df.iloc[x-2, :]['Speed 2'])
            frame.append((rs1 + rs2)/2)
        
        frame.append((df.iloc[x, :]['Distance 1'] - df.iloc[x, :]['Distance 2']))
        frame.append((df.iloc[x, :]['Speed 1'] - df.iloc[x, :]['Speed 2']) * (df.iloc[x, :]['Distance 1'] - df.iloc[x, :]['Distance 2']))
        
        if df.iloc[x, :]['Vehicle no'] == 8:
            frame.extend(list(df.iloc[x, :][['Camera', 'Lidar', 'Radar']]))
        
        
    data['C'] = data['C'].astype(int)
    data['L'] = data['L'].astype(int)
    data['R'] = data['R'].astype(int)
    
    data['CLR'] = (4*data['C'] + 2*data['L'] + data['R']) - 1
    
    data.to_excel(filename_car_processed, index = False)
    return df, data

def adam_optimizer():
    return tf.train.AdamOptimizer(
                            learning_rate=tf.compat.v1.train.exponential_decay(
                                    learning_rate=0.1,
                                    global_step=tf.compat.v1.train.get_global_step(),
                                    decay_steps=1000,
                                    decay_rate=0.96)
                            )

def model_spec_classifier(train_df, validate_df, target_columns, n_classes = 2, feature_columns = features):
#    pdb.set_trace()
    feature_columns_df = [tf.feature_column.numeric_column(key) for key in feature_columns]
    
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns_df,
                                            hidden_units = hidden_units_spec,
                                            dropout = dropout,
                                            n_classes=n_classes,
                                            #optimizer= adam_optimizer,
                                            batch_norm=batch_norm
                                            )#, n_classes=n_classes_spec, model_dir=tmp_dir_spec)
    
    # Train the model:
    train_input_fn = tf.estimator.inputs.pandas_input_fn(x=train_df[feature_columns], 
                                                         y = pd.Series(train_df[target_columns[0]], index=train_df.index, dtype='int32'), 
                                                         num_epochs=epochs_spec, 
                                                         shuffle=False)
    classifier.train(input_fn=train_input_fn, steps=steps_spec)
    
    #Validate the model:
    validate_input_fn = tf.estimator.inputs.pandas_input_fn(x=validate_df[feature_columns], 
                                                            y=pd.Series(validate_df[target_columns[0]], index = validate_df.index, dtype = 'int32'), 
                                                            num_epochs=epochs_spec, 
                                                            shuffle=False)

    accuracy_score_m = classifier.evaluate(input_fn=validate_input_fn)["accuracy"]
    print("Accuracy = {}".format(accuracy_score_m))

    return classifier

def model_spec_regressor(train_df, validate_df, target):
    '''
    Applying the DNN regressor here, but cannot apply different activation function on different layers.
    Thus, moved to hand built controllabe Keras regressor
    '''
    feature_columns_df = [tf.feature_column.numeric_column(key) for key in features]
    
    regressor = tf.estimator.DNNRegressor(feature_columns=feature_columns_df, 
                                          hidden_units = hidden_units_spec,
                                          activation_fn = tf.nn.relu)#, n_classes=n_classes_spec, model_dir=tmp_dir_spec)
    
    # Train the model:
    train_input_fn = tf.estimator.inputs.pandas_input_fn(x          = train_df[features], 
                                                         y          = pd.Series(train_df[target[0]], train_df.index, dtype = 'float'), 
                                                         num_epochs = epochs_spec, 
                                                         shuffle    = False)
    regressor.train(input_fn=train_input_fn, steps=steps_spec)
    
    #Validate the model:
    validate_input_fn = tf.estimator.inputs.pandas_input_fn(x          = validate_df[features], 
                                                            y          = pd.Series(validate_df[target[0]], validate_df.index, dtype = 'float'), 
                                                            num_epochs = epochs_spec, 
                                                            shuffle    = False)


    
    evaluation_result = regressor.evaluate(input_fn=validate_input_fn)
    print("Average Loss = {}".format(evaluation_result['average_loss']))

    return regressor

def result_preparation(df):
    df['ABS_Error'] = abs(df.ix[:, 0] - df.ix[:, 1])
    df['SQR_Error'] = df['ABS_Error'] * df['ABS_Error']
    
    return df

def prediction(classifier, test_df, target, do_print = True, do_excel = True):
     test_pred = test_df.drop(labels=target[0], axis=1)
     predict_input_fn = tf.estimator.inputs.pandas_input_fn(x=test_pred, 
                                                               num_epochs=1, 
                                                               shuffle=False)
          
     targets = test_df[target[0]].tolist()
     df = pd.DataFrame()
     df['Target'] = targets

     if str(type(classifier)).find('DNNClassifier') > 0:
         filename = output_files[0]
     
         predictions = list(classifier.predict(input_fn=predict_input_fn))
         predicted_classes = [p["probabilities"][1] for p in predictions]  
         predicted_labels = [p["classes"] for p in predictions] 
     
         df['Prediction'] = predicted_classes
         df['Class'] = [ 1 if x > 0.5 else 0 for x in predicted_classes]
         
#         if do_print == True:
#             for t, p in zip(targets, predicted_labels):
#                 print(t, p)     
     
     else:
         filename = output_files[1]
         
         predictions = classifier.predict(input_fn=predict_input_fn)
         predicted_output = []
         
         for x in list(predictions):
             predicted_output = predicted_output + list(x['predictions'])  
         
         df['Prediction'] = predicted_output
         if do_print == True:
             for t, p in zip(targets, predicted_output):
                 print(t, p)
         #pdb.set_trace()
     
         df = result_preparation(df)
    
     df.to_excel(filename, index = False)
     
     return predictions, df
 
    
def keras_regression(train_data, test_data, target, features):
    x_train = train_data[features]
    y_train = train_data[target[0]]
    input_dict = {
                    'hidden_layers' : hidden_units_spec,
                    'input_dm' : 3,
                    'activation_fn': ['relu', 'relu', 'relu', 'sigmoid'],
                    'k_fold': 0, # <=0 : for no kfold cross-validation, >0: do kfild validation
                    'loss' : 'mean_squared_error',
                    'optimizer' : 'adam',
                    'model_type':'regressor',
                    'no_of_output' : 1,
                    'metrics' : ['mse'],
                    'dropout_spec': dropout_spec,
                    'sample_weight' : None #np.array(sample_weight)
                 } 
    
    
    Deep_Model = DM.Deep_Models("Deep Neural Network")
    print(Deep_Model.get_name())
    regressor = Deep_Model.DNN_Models(x_train, y_train, **input_dict)
        
    x_test = test_data[features]
    y_test = test_data[target[0]]
    
    predicted_output = regressor.predict(x_test)
    
    for t, p in zip(y_test, predicted_output):
        print(t, p)
    
    df = pd.DataFrame()
    df['Target'] = y_test
    df['Prediction'] = predicted_output
    
    df = result_preparation(df)
    df.to_excel(filename_reg, index = False)
    
    return regressor

def keras_MOClassifier(train_data, test_data, target):
    x_train = train_data[features]
    y_train = train_data[target]
    
    sample_weight = read_sample_weight(filename, 'Sample_Weight', sheet_name[2])
    sample_weight = sample_weight.iloc[list(train_data.index)]
    input_dict = {
                'hidden_layers' : hidden_units_spec,
                'input_dm' : 3,
                'activation_fn': ['relu', 'relu', 'relu', 'sigmoid'],  # should be sigmoid at the end according to https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/
                'k_fold': 0, # <=0 : for no kfold cross-validation, >0: do kfild validation
                'loss' : 'binary_crossentropy',  # treat each output as a binary classifier - thus ensure to trigger the output indipendantly from others
                'optimizer' : 'adam',
                'model_type':'classifier',
                'no_of_output' : 6,
                'metrics' : ['accuracy'],
                'dropout_spec': dropout_spec,
                'sample_weight' : None#np.array(sample_weight)
             } 
    
    Deep_Model = DM.Deep_Models("Deep Neural Network - Multi-output Classifier")
    print(Deep_Model.get_name())
    MOClassifier = Deep_Model.DNN_Models(x_train, y_train, **input_dict)

    x_test = test_data[features]
    y_test = test_data[target].reset_index(drop = True)
    
    predicted_probability = MOClassifier.predict_proba(x_test)
    df_pred = pd.DataFrame([], columns = list(x + '_Pred' for x in list(y_test.columns)))
    
    for pred in predicted_probability:
        df_row = pd.DataFrame([pred], columns = df_pred.columns)
        df_pred = pd.concat([df_pred, df_row], ignore_index = True, axis = 0)            
    
        
    df_class = df_pred.gt(0.5).astype(int)
    df_class.columns = list(x + '_Pred_Class' for x in list(y_test.columns))

    df = pd.concat([y_test, df_pred, df_class], axis = 1)
    df.to_excel(filename_MOC, index = False)

    return MOClassifier, df


def keras_MOClassifier_car(train_data, test_data, target_columns, feature_columns):
    x_train = train_data[feature_columns]
    y_train = train_data[target_columns]

    input_dict = {
            'hidden_layers' : hidden_units_spec,
            'input_dm' : len(feature_columns),
            'activation_fn': ['relu', 'relu', 'relu', 'sigmoid'],  # should be sigmoid at the end according to https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/
            'k_fold': 0, # <=0 : for no kfold cross-validation, >0: do kfold validation
            'loss' : 'binary_crossentropy',  # treat each output as a binary classifier - thus ensure to trigger the output indipendantly from others
            'optimizer' : 'adam',
            'model_type':'classifier',
            'no_of_output' : len(target_columns),
            'metrics' : ['accuracy'],
            'dropout_spec': dropout_spec,
            'sample_weight' : None#np.array(sample_weight)
         } 
    Deep_Model = DM.Deep_Models("Deep Neural Network - Multi-output Classifier")
    print(Deep_Model.get_name())
    MOClassifier = Deep_Model.DNN_Models(x_train, y_train, **input_dict)

    
    x_test = test_data[feature_columns]
    y_test = test_data[target_columns].reset_index(drop = True)

        
    if len(list(y_test.columns)) > 1:
        predicted_probability = MOClassifier.predict_proba(x_test)
    else: # for single class classifier, we need one output column
        predicted_probability = MOClassifier.predict(x_test)
    
    
    df_pred = pd.DataFrame([], columns = list(x + '_Pred' for x in list(y_test.columns)))
   
 #   pdb.set_trace()
    for pred in predicted_probability:
        df_row = pd.DataFrame([pred], columns = df_pred.columns)
        df_pred = pd.concat([df_pred, df_row], ignore_index = True, axis = 0)  

    df_class = df_pred.gt(0.5).astype(int)
    df_class.columns = list(x + '_Pred_Class' for x in list(y_test.columns))
        
    df = pd.concat([y_test, df_pred, df_class], axis = 1)
    
    df.to_excel(filename_MOC_car, index = False)
    
    return MOClassifier, df

def keras_BClassifier_Car(train_data, test_data, target_columns, feature_columns):
    x_train = train_data[feature_columns]
    y_train = train_data[target_columns]
    
    input_dict = {
                'hidden_layers' : hidden_units_spec,
                'input_dm' : len(feature_columns),
                'activation_fn': ['relu', 'relu', 'relu', 'sigmoid'],  # should be sigmoid at the end according to https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/
                'k_fold': 0, # <=0 : for no kfold cross-validation, >0: do kfild validation
                'loss' : 'binary_crossentropy',  # treat each output as a binary classifier - thus ensure to trigger the output indipendantly from others
                'optimizer' : 'adam',
                'model_type':'classifier',
                'no_of_output' : len(target_columns),
                'metrics' : ['accuracy'],
                'dropout_spec': dropout_spec,
#                'class_weight': {0: 1., 1: 1.},
                'sample_weight' : None#np.array(sample_weight)
             } 
    
    Deep_Model = DM.Deep_Models("Deep Neural Network - Binary Classifier")
    print(Deep_Model.get_name())
#    BClassifier = Deep_Model.DNN_Models_Class_Weight(x_train, y_train, **input_dict)
    BClassifier = Deep_Model.DNN_Models(x_train, y_train, **input_dict)

    x_test = test_data[feature_columns]
    y_test = test_data[target_columns].reset_index(drop = True)
    
    predicted_probability = BClassifier.predict(x_test)
        
    df = pd.concat([y_test, pd.DataFrame(predicted_probability, columns = ['Predicted_Class'])], axis = 1)
    df.to_excel(filename_BC_car, index = False)

    return BClassifier, df

def keras_BClassifier(train_data, test_data, target):
    
    x_train = train_data[features]
    y_train = train_data[target]
    
    input_dict = {
                'hidden_layers' : hidden_units_spec,
                'input_dm' : 3,
                'activation_fn': ['relu', 'relu', 'relu', 'sigmoid'],  # should be sigmoid at the end according to https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/
                'k_fold': 0, # <=0 : for no kfold cross-validation, >0: do kfild validation
                'loss' : 'binary_crossentropy',  # treat each output as a binary classifier - thus ensure to trigger the output indipendantly from others
                'optimizer' : 'adam',
                'model_type':'classifier',
                'no_of_output' : 1,
                'metrics' : ['accuracy'],
                'dropout_spec': dropout_spec,
                'class_weight': {0: 50.,
                                 1: 1.},
                'sample_weight' : None#np.array(sample_weight)
             } 
    
    Deep_Model = DM.Deep_Models("Deep Neural Network - Binary Classifier")
    print(Deep_Model.get_name())
    BClassifier = Deep_Model.DNN_Models_Class_Weight(x_train, y_train, **input_dict)

    x_test = test_data[features]
    y_test = test_data[target].reset_index(drop = True)
    
    predicted_class = BClassifier.predict(x_test)
    predicted_probability = BClassifier.predict_proba(x_test)

    df = pd.concat([y_test,  pd.DataFrame(predicted_probability[:,1], columns = ['Prediction']), pd.DataFrame(predicted_class, columns = ['Predicted_Class'])], axis = 1)
    df.to_excel(filename_BC, index = False)

    return BClassifier, df

def Keras_classifier_all_data():
    
    df_tr = read_data(filename, features, target, sheet_name[0])    # Read the trustworthy data
    df_utr = read_data(filename, features, target, sheet_name[1])    # Read the untrustworthy data
    df_combined = prepare_data(df_tr, df_utr, True)                 # combining the two data set
    
    # divide the dataset in to train, validate and test - training, validating and predicting.
    
    train_df, test_df = np.split(df_combined.sample(frac=1), [int(.8*len(df_combined))])    
    
    classifier, df = keras_BClassifier(train_df, test_df, target)
    #classifier = model_spec_regressor(train_df, validate_df)    
 #   _, df = prediction(classifier, test_df, target, do_print = True)
    
    df.columns = ['Target', 'Prediction', 'Class']
#    if accuracy_score(df.iloc[:,0], df.iloc[:,0+2]) < 0.70:
#        return 0, 0, 0
    
    Model_Eval = ME.Model_Evaluation("Keras_Multiclass_Classifier_Car")
    #    pdb.set_trace()
    Model_Eval.metrics_printer(df, len(target), 2)
    Model_Eval.ROC_Curve_Generator(df, len(target), 1) # Ofset to the real probability not the class level, see the df and the generated excel file

    Model_Eval.metrics_file_writer(metrics_file_car, df, len(target), 2)
    # list(data.columns).index('C') this is the offset of output labels in the dataset
    Model_Eval.PR_Curve_Generator(df, len(target), 1, df_combined, list(df_combined.columns).index('Two_Class')) # adjust to 3 when phasetime will be added as feature set
    
        
    return classifier, test_df, df

def DNN_classifier_all_data():
    
    df_tr = read_data(filename, features, target, sheet_name[0])    # Read the trustworthy data
    df_utr = read_data(filename, features, target, sheet_name[1])    # Read the untrustworthy data
    df_combined = prepare_data(df_tr, df_utr, True)                 # combining the two data set
    
    # divide the dataset in to train, validate and test - training, validating and predicting.
    
    train_df, validate_df, test_df = np.split(df_combined.sample(frac=1), [int(.6*len(df_combined)), int(.8*len(df_combined))])    
    
    classifier = model_spec_classifier(train_df, validate_df, target, n_classes)
    #classifier = model_spec_regressor(train_df, validate_df)    
    _, df = prediction(classifier, test_df, target, do_print = True)
    
    
    if accuracy_score(df.iloc[:,0], df.iloc[:,0+2]) < 0.70:
        return 0, 0, 0
    
    Model_Eval = ME.Model_Evaluation("Keras_Multiclass_Classifier_Car")
    #    pdb.set_trace()
    Model_Eval.metrics_printer(df, len(target), 2)
    Model_Eval.ROC_Curve_Generator(df, len(target), 1) # Ofset to the real probability not the class level, see the df and the generated excel file

    Model_Eval.metrics_file_writer(metrics_file_car, df, len(target), 2)
    # list(data.columns).index('C') this is the offset of output labels in the dataset
    Model_Eval.PR_Curve_Generator(df, len(target), 1, df_combined, list(df_combined.columns).index('Two_Class')) # adjust to 3 when phasetime will be added as feature set
    
        
    return classifier, test_df, df


def DNN_classifier_untrustworthy():
    
    df_utr = read_data(filename, features, target_CF, sheet_name[1])    # Read the untrustworthy data
    
    # divide the dataset in to train, validate and test - training, validating and predicting.
    
    train_df, validate_df, test_df = np.split(df_utr.sample(frac=1), [int(.6*len(df_utr)), int(.8*len(df_utr))])    
    
    classifier = model_spec_classifier(train_df, validate_df, target_CF, n_classes)
    #classifier = model_spec_regressor(train_df, validate_df)    
    predictions = prediction(classifier, test_df, target_CF, do_print = True)
    
    return classifier, test_df, predictions

def DNN_multiclass_classifier():
    '''
    This section use DNN classifier to predict multi-level multi-class classifier, not yet successful
    The user of 1, 2, 3, 4, 5, 6 as class lable is fundamentally flawed, as 1 faulty sensor cannot differentiate
    between which sensor is faulty
    '''
    df_utr_MC = read_data(filename, features, target_MC, sheet_name[1])    # Reading untrustworthy data 
    train_df_MC, validate_df_MC, test_df_MC = np.split(df_utr_MC.sample(frac=1), [int(.6*len(df_utr_MC)), int(.8*len(df_utr_MC))])     
    classifier_MC = model_spec_classifier(train_df_MC, validate_df_MC, target_MC, n_classes_MC)   
    
    predictions = prediction(classifier_MC, test_df_MC, target_MC, do_print = True, do_excel = False)
    
    return classifier_MC, test_df_MC, predictions

def DNN_multiclass_classifier_Car():
    '''
    This one will generate a multi-level classifier that convert the problem to single class.
    '''
    data_excel, data = read_data_car(filename_car, data_car_columns, sheet_name = 'data_car')
#    columns = ['A1', 'D1', 'SD1', 'A2', 'D2', 'SD2', 'A3', 'RS3', 'D3', 'SD3', 'A4', 'RS4', 'D4', 'SD4',
#               'A5', 'RS5', 'D5', 'SD5', 'A6', 'RS6', 'D6', 'SD6', 'A7', 'RS7', 'D7', 'SD7', 'A8', 'RS8', 'D8', 'SD8',
#               'C', 'L', 'R', 'CLR']

    data = data[['A8', 'RS8', 'D8', 'SD8', 'CLR']] # when we do not need all columns to consider

    target_columns = ['CLR']  

    feature_columns = [x for x in list(data.columns) if x not in target_columns]   
        
    train_data, validate_data, test_data = np.split(data.sample(frac=1), [int(.6*len(data)), int(.8*len(data))])     
    
    classifier_MCC = model_spec_classifier(train_data, validate_data, target_columns, 7, feature_columns)
    
    predictions = prediction(classifier_MCC, test_data, target_columns, do_print = True, do_excel = False)
    
    return classifier_MCC, test_data, predictions

def DNN_regressor():
    '''
    This section apply DNN defined regressors
    '''
    df_utr_Reg = read_data(filename, features, target_Reg, sheet_name[1])   # reading only untrustworthy Data
    train_df_Reg, validate_df_Reg, test_df_Reg = np.split(df_utr_Reg.sample(frac=1), [int(.6*len(df_utr_Reg)), int(.8*len(df_utr_Reg))])    
    
    classifier_Reg = model_spec_regressor(train_df_Reg, validate_df_Reg, target_Reg)    
    predictions = prediction(classifier_Reg, test_df_Reg, target_Reg, do_print = True, do_excel = False)
    
    return classifier_Reg, test_df_Reg, predictions

def keras_regression_fn():
    
    ''' 
    This section only consider the  untrustworthy data, and predict the continuous value of trustworthiness.
   
    '''
    df_utr_Reg = read_data(filename, features, target_Reg, sheet_name[1])  # reading untrustworthy data  
    train_df_Reg, test_df_Reg = np.split(df_utr_Reg.sample(frac=1), [int(.6*len(df_utr_Reg))])    
    regressor = keras_regression(train_df_Reg, test_df_Reg, target_Reg)
    
    return regressor, test_df_Reg

def Keras_multiclass_classifier():
    '''
    This section considers untrustworthy data and a multilevel - multiclass classifier.
    '''
    df_utr_MOC = read_data(filename, features, target_MO, sheet_name[2])      # reading untrustworthy multiclass data
    train_df_MOC, test_df_MOC = np.split(df_utr_MOC.sample(frac=1), [int(.6*len(df_utr_MOC))])    
    MOClassifier, df = keras_MOClassifier(train_df_MOC, test_df_MOC, target_MO)
    
    Model_Eval = ME.Model_Evaluation("Keras_Multiclass_Classifier")
    
    Model_Eval.metrics_printer(df, 6, 12) # 6 outputs to measure the accuracy for and 12 is the offset to find the predicted column for a true label
    Model_Eval.ROC_Curve_Generator(df, 6, 6) # Ofset to the real probability not the class level, see the df and the generated excel file
    Model_Eval.metrics_file_writer(metrics_file, df, 6, 12)
    Model_Eval.PR_Curve_Generator(df, 6, 6, df_utr_MOC, 3) # adjust to 3 when phasetime will be added as feature set
    
    return MOClassifier, test_df_MOC, df

def Keras_multiclass_cassifier_car():
    
    data_excel, data = read_data_car(filename_car, data_car_columns, sheet_name = 'data_car')
#    columns = ['A1', 'D1', 'SD1', 'A2', 'D2', 'SD2', 'A3', 'RS3', 'D3', 'SD3', 'A4', 'RS4', 'D4', 'SD4',
#               'A5', 'RS5', 'D5', 'SD5', 'A6', 'RS6', 'D6', 'SD6', 'A7', 'RS7', 'D7', 'SD7', 'A8', 'RS8', 'D8', 'SD8',
#               'C', 'L', 'R']

#    data = data[['A8', 'RS8', 'D8', 'SD8', 'C', 'L', 'R']] # when we do not need all columns to consider

    # this line needed if only considered mutually exclusive output
    #data = data[data.C + data.L + data.R == 1]

    train_data, test_data = np.split(data.sample(frac=1), [int(0.7*len(data))])
    
    feature_columns = [x for x in list(data.columns) if x not in target_car]
    
    MOClassifier, df = keras_MOClassifier_car(train_data, test_data, target_car, feature_columns)
    
    Model_Eval = ME.Model_Evaluation("Keras_Multiclass_Classifier_Car")
    
#    pdb.set_trace()
    Model_Eval.metrics_printer(df, len(target_car), 6)
    Model_Eval.ROC_Curve_Generator(df, len(target_car), 3) # Ofset to the real probability not the class level, see the df and the generated excel file

    Model_Eval.metrics_file_writer(metrics_file_car, df, len(target_car), 6)
    # list(data.columns).index('C') this is the offset of output labels in the dataset
    Model_Eval.PR_Curve_Generator(df, len(target_car), 3, data, list(data.columns).index('C')) # adjust to 3 when phasetime will be added as feature set
    
    return MOClassifier, data_excel, data, test_data, df


def Keras_binary_classifier():
    '''
    This section considers untrustworthy data and a multilevel - multiclass classifier.
    '''
    df_utr_BC = read_data(filename, features, target_CF, sheet_name[1])      # reading untrustworthy multiclass data
    train_df_BC, test_df_BC = np.split(df_utr_BC.sample(frac=1), [int(.6*len(df_utr_BC))])    
    BClassifier, df = keras_BClassifier(train_df_BC, test_df_BC, target_CF)
    
    return BClassifier, test_df_BC     

def Binary_classifier_car(DNN_Keras = 'Keras'):
# *************  Data Reading and preparation block **************
    data_excel, data = read_data_car(filename_car, data_car_columns, sheet_name = 'data_car')   

#    columns = ['A1', 'D1', 'SD1', 'A2', 'D2', 'SD2', 'A3', 'RS3', 'D3', 'SD3', 'A4', 'RS4', 'D4', 'SD4',
#               'A5', 'RS5', 'D5', 'SD5', 'A6', 'RS6', 'D6', 'SD6', 'A7', 'RS7', 'D7', 'SD7', 'A8', 'RS8', 'D8', 'SD8',
#               'C', 'L', 'R']    
#    data = data[['A8', 'RS8', 'D8', 'SD8', 'C', 'L', 'R']] # when we do not need all columns to consider    
# this line needed if only considered mutually exclusive output
#    data = data[data.C + data.L + data.R == 1]    

    data = data[[ 'A8', 'RS8', 'D8', 'SD8',
               'L']] # when we do not need all columns to consider #Change the target avobe and below - C, L or R based on which target to be used in the binary classifier    
    target_columns = ['L']   
    feature_columns = [x for x in list(data.columns) if x not in target_columns]          
    
# divide the dataset in to train, validate and test - training, validating and predicting.
    if DNN_Keras == 'Keras':
        train_data, test_data = np.split(data.sample(frac=1), [int(0.8*len(data))])
        
        BClassifier_Car, df = keras_BClassifier_Car(train_data, test_data, target_columns, feature_columns) 

    elif DNN_Keras == 'DNN':
        train_data, validate_data, test_data = np.split(data.sample(frac=1), [int(.6*len(data)), int(.8*len(data))])   
        BClassifier_Car = model_spec_classifier(train_data, validate_data, target_columns, 2, feature_columns)
    
        _, df = prediction(BClassifier_Car, test_data, target_columns, do_print = True)
        df.Prediction = df.Prediction.gt(0.5).astype(int)
        
#    pdb.set_trace()    
    Model_Eval = ME.Model_Evaluation("Keras_Multiclass_Classifier_Car")
    Model_Eval.metrics_printer(df, len(target_columns), 1)
    Model_Eval.ROC_Curve_Generator(df, len(target_columns), 1)

    Model_Eval.PR_Curve_Generator(df, len(target_columns), 1, data, list(data.columns).index('L'))
        
    return BClassifier_Car, data_excel, data, df
       
if __name__ == '__main__':
    print('Hi from the main')
    #df, data = read_data_car(filename_car, data_car_columns, sheet_name = 'data_car')
    
#    MOClassifier, data_excel, data_processed, test_df_MOC, df = Keras_multiclass_cassifier_car()
 #   DNN_multiclass_classifier_Car()    
#    while True:
#    classifier, test_df, df = DNN_classifier_all_data()
#        if type(classifier) == int:
#            continue
#        else: 
#            break 


#    Keras_classifier_all_data()
#    BClassifier, data_excel, data_processed, df = Binary_classifier_car('Keras')
#    Keras_model_tuner_Binary_Car()
#    regressor, test_df_Reg = keras_regression_fn()
#     MOClassifier, test_df_MOC, df = Keras_multiclass_classifier()
#     Keras_model_tuner()
#     DNN_classifier_untrustworthy()
#    BClassifier, test_df_BC  = Keras_binary_classifier()

    


import csv
import numpy as np

# labels
#  0. silence
#  1. unknown
#  2. yes
#  3. no
#  4. up
#  5. down
#  6. left
#  7. right
#  8. on
#  9. off
# 10. stop
# 11. go

class_labels = ('silence','unknown','yes','no','up','down','left','right','on','off','stop','go')

########################################################################
# This code assumes all source files have been sorted in the same way. #
########################################################################


# with open('pred_all.csv', 'w') as csvfile:
#with open("../test_ensemble_DSCNN_GRU_LSTM.csv", 'w') as ensemble_csv:
# with open("../test_ensemble_DSCNN_GRU_CRNN.csv", 'w') as ensemble_csv:
# with open("../test_ensemble_DSCNN_GRU_LSTM_excluding_unknown.csv", 'w') as ensemble_csv:
# with open("../test_ensemble_DSCNN_GRU_CRNN_excluding_unknown.csv", 'w') as ensemble_csv:
#with open("./test_ensemble_normal_cnn_GRU_CRNN.csv", 'w') as ensemble_csv:
#with open("./test_ensemble_DS_CNN3_up_0.50_GRU_CRNN.csv", 'w') as ensemble_csv:
#with open("./test_ensemble_normal_CNN_up_0.50_GRU_CRNN.csv", 'w') as ensemble_csv:
#with open("./test_ensemble_normal_CNN_up_0.50_GRU_LSTM.csv", 'w') as ensemble_csv:
#with open("./test_ensemble_normal_CNN_up_0.50_DS_CNN3_up_0.60_GRU.csv", 'w') as ensemble_csv:
#with open("./test_ensemble_normal_CNN_up_0.60_finer_input_DS_CNN3_up_0.60_GRU.csv", 'w') as ensemble_csv:
#with open("./test_ensemble_normal_CNN_up_0.60_finer_input_GRU_CRNN.csv", 'w') as ensemble_csv:
#with open("./test_ensemble_top3_ensembles.csv", 'w') as ensemble_csv:
#with open("./test_ensemble_normal_CNN_7l_512ch_20dctcc_10winstride_0.50up_0.50dropout_GRU3_up_0.50_CRNN3_up_0.50.csv", 'w') as ensemble_csv:
#with open("./test_ensemble_current_top_3s_180112_0009.csv", 'w') as ensemble_csv:
# with open("./test_ensemble_current_top_3s_180112_2242.csv", 'w') as ensemble_csv:
#with open("./test_ensemble_current_top_5s_180115_0631.csv", 'w') as ensemble_csv:
#with open("./test_ensemble_180116_0633.csv", 'w') as ensemble_csv:
#with open("./test_ensemble_180116_0641.csv", 'w') as ensemble_csv:
#with open("./test_ensemble_180116_2302.csv", 'w') as ensemble_csv:
#with open("./test_ensemble_180116_2307.csv", 'w') as ensemble_csv:
with open("./test_ensemble_final_180117.csv", 'w') as ensemble_csv:
  ensemble_csv_wr = csv.writer(ensemble_csv, delimiter=',',
                           quotechar='|', quoting=csv.QUOTE_MINIMAL)
  # # Ensemble 0
  # pred_csv_0 = open("test_DS-CNN_3.csv", "r")
  # pred_csv_1 = open("test_GRU_L.csv", "r")
  # pred_csv_2 = open("test_LSTM_L.csv", "r")
  ## Ensemble 1
  #pred_csv_0 = open("../test_DS-CNN_3.csv", "r")
  #pred_csv_1 = open("../test_GRU_L.csv", "r")
  #pred_csv_2 = open("../test_CRNN_L.csv", "r")
  ## Ensemble 2
  #pred_csv_0 = open("./test_normal_cnn.csv", "r")
  #pred_csv_1 = open("./test_GRU_L.csv", "r")
  #pred_csv_2 = open("./test_CRNN_L.csv", "r")
  ## Ensemble 3
  #pred_csv_0 = open("./test_DS_CNN3_up_0.50.csv", "r")
  #pred_csv_1 = open("./test_GRU_L.csv", "r")
  #pred_csv_2 = open("./test_CRNN_L.csv", "r")
  ## Ensemble 4
  #pred_csv_0 = open("./test_normal_CNN_up_0.50.csv", "r")
  #pred_csv_1 = open("./test_GRU_L.csv", "r")
  #pred_csv_2 = open("./test_CRNN_L.csv", "r")
  ## Ensemble 4
  #pred_csv_0 = open("./test_normal_CNN_up_0.50.csv", "r")
  #pred_csv_1 = open("./test_GRU_L.csv", "r")
  #pred_csv_2 = open("./test_LSTM_L.csv", "r")
  ## Ensemble 4
  #pred_csv_0 = open("./test_normal_CNN_up_0.50.csv", "r")
  #pred_csv_1 = open("./test_DS_CNN3_up_0.60.csv", "r")
  #pred_csv_2 = open("./test_GRU_L.csv", "r")
  ## Ensemble 4
  #pred_csv_0 = open("./test_normal_CNN_up_0.60_finer_input.csv", "r")
  #pred_csv_1 = open("./test_DS_CNN3_up_0.60.csv", "r")
  #pred_csv_2 = open("./test_GRU_L.csv", "r")
  ## Ensemble 4
  #pred_csv_0 = open("./test_normal_CNN_up_0.60_finer_input.csv", "r")
  #pred_csv_1 = open("./test_GRU_L.csv", "r")
  #pred_csv_2 = open("./test_CRNN_L.csv", "r")
  ## Ensemble 4
  #pred_csv_0 = open("./test_ensemble_normal_CNN_up_0.60_finer_input_DS_CNN3_up_0.60_GRU.csv", 'r')
  #pred_csv_1 = open("./test_ensemble_normal_CNN_up_0.60_finer_input_GRU_CRNN.csv", 'r')
  #pred_csv_2 = open("./test_ensemble_normal_CNN_up_0.50_DS_CNN3_up_0.60_GRU.csv", 'r')
  ## Ensemble 4
  #pred_csv_0 = open("./test_normal_CNN_7l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  #pred_csv_1 = open("./test_GRU3_up_0.50.csv", 'r')
  #pred_csv_2 = open("./test_CRNN3_up_0.50.csv", 'r')
  ## Ensemble 4
  #pred_csv_0 = open("./test_normal_CNN_7l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  #pred_csv_1 = open("./test_normal_CNN_7l_512ch_40dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  #pred_csv_2 = open("./test_normal_CNN_10l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  ## Ensemble 4 <<<<<<<<<< Current best >>>>>>>>>>>>
  #pred_csv_0 = open("./test_normal_CNN_11l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  #pred_csv_1 = open("./test_normal_CNN_7l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  #pred_csv_2 = open("./test_normal_CNN_7l_512ch_40dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  ## Ensemble 4
  #  pred_csv_0 = open("./test_normal_CNN_11l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  #  pred_csv_1 = open("./test_normal_CNN_7l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  #  pred_csv_2 = open("./test_normal_CNN_7l_512ch_40dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  #  pred_csv_3 = open("./test_normal_CNN_10l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  #  pred_csv_4 = open("./test_normal_CNN_14l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  ## Ensemble test_ensemble_180116_0633.csv
  #pred_csv_0 = open("./test_ensemble_current_top_3s_180112_2242.csv", 'r')
  #pred_csv_1 = open("./test_normal_CNN_11l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  #pred_csv_2 = open("./test_normal_CNN_10l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  #pred_csv_3 = open("./test_normal_CNN_13l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  #pred_csv_4 = open("./test_normal_CNN_14l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  ## Ensemble test_ensemble_180116_0641.csv
  #pred_csv_0 = open("./test_normal_CNN_11l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  #pred_csv_1 = open("./test_normal_CNN_11l_512ch_20dctcc_10winstride_0.50up_0.50dropout_copy1.csv", 'r')
  #pred_csv_2 = open("./test_normal_CNN_11l_512ch_20dctcc_10winstride_0.50up_0.50dropout_copy2.csv", 'r')
  #pred_csv_3 = open("./test_normal_CNN_7l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  #pred_csv_4 = open("./test_normal_CNN_7l_512ch_40dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  ## Ensemble "./test_ensemble_180116_2302.csv"
  #pred_csv_0 = open("./test_normal_CNN_11l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  #pred_csv_1 = open("./test_normal_CNN_7l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  #pred_csv_2 = open("./test_normal_CNN_7l_512ch_40dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  ## Ensemble "./test_ensemble_180116_2307.csv"
  #pred_csv_0 = open("./test_normal_CNN_11l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  #pred_csv_1 = open("./test_normal_CNN_7l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  #pred_csv_2 = open("./test_normal_CNN_7l_512ch_40dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  # Ensemble "./test_ensemble_final_180117.csv"
  pred_csv_0 = open("./test_normal_CNN_11l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  pred_csv_1 = open("./test_normal_CNN_11l_512ch_40dctcc_10winstride_0.50up_0.50dropout.csv", 'r')
  pred_csv_2 = open("./test_normal_CNN_7l_512ch_20dctcc_10winstride_0.50up_0.50dropout.csv", 'r')


  ##################################################################################
  # (1) Ensemble from 3 predictions
  ##################################################################################
  # Remove header
  pred_csv_0.readline()
  pred_csv_1.readline()
  pred_csv_2.readline()

  pred_csv_rd_0 = csv.reader(pred_csv_0)
  pred_csv_rd_1 = csv.reader(pred_csv_1)
  pred_csv_rd_2 = csv.reader(pred_csv_2)

  ensemble_csv_wr.writerow(['fname', 'label'])

  # Same priority between classes
  for row_0, row_1, row_2 in zip(pred_csv_rd_0, pred_csv_rd_1, pred_csv_rd_2):
    # Simple majority voting from three predictions
    # The highest priority is 'DSCNN'.
    if row_1[1]==row_2[1]:
      sel_class = row_1[1]
    else:
      sel_class = row_0[1]
    
    ensemble_csv_wr.writerow([row_0[0], sel_class])

  ## Unknown-weighted ensemble
  #for row_0, row_1, row_2 in zip(pred_csv_rd_0, pred_csv_rd_1, pred_csv_rd_2):
  #  # Simple majority voting from three predictions
  #  # The highest priority is 'DSCNN'.
  #  if (row_0[1]=='unknown') or (row_1[1]=='unknown')  or (row_2[1]=='unknown'):
  #    sel_class = 'unknown'
  #  else:
  #    if row_1[1]==row_2[1]:
  #      sel_class = row_1[1]
  #    else:
  #      sel_class = row_0[1]
  #  
  #  ensemble_csv_wr.writerow([row_0[0], sel_class])

  # ##################################################################################
  # # (2) Ensemble from 5 predictions
  # ##################################################################################
  # # Remove header
  # pred_csv_0.readline()
  # pred_csv_1.readline()
  # pred_csv_2.readline()
  # pred_csv_3.readline()
  # pred_csv_4.readline()

  # pred_csv_rd_0 = csv.reader(pred_csv_0)
  # pred_csv_rd_1 = csv.reader(pred_csv_1)
  # pred_csv_rd_2 = csv.reader(pred_csv_2)
  # pred_csv_rd_3 = csv.reader(pred_csv_3)
  # pred_csv_rd_4 = csv.reader(pred_csv_4)

  # ensemble_csv_wr.writerow(['fname', 'label'])

  # for row_0, row_1, row_2, row_3, row_4 in zip(pred_csv_rd_0, pred_csv_rd_1, pred_csv_rd_2, pred_csv_rd_3, pred_csv_rd_4):
  #   # Majority vote from 5 predictions
  #   class_cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  #   class_cnt[class_labels.index(row_0[1])] = class_cnt[class_labels.index(row_0[1])] + 1
  #   class_cnt[class_labels.index(row_1[1])] = class_cnt[class_labels.index(row_1[1])] + 1
  #   class_cnt[class_labels.index(row_2[1])] = class_cnt[class_labels.index(row_2[1])] + 1
  #   class_cnt[class_labels.index(row_3[1])] = class_cnt[class_labels.index(row_3[1])] + 1
  #   class_cnt[class_labels.index(row_4[1])] = class_cnt[class_labels.index(row_4[1])] + 1

  #   max_cnt_class_num = np.argmax(class_cnt)
  #   if (class_cnt[max_cnt_class_num]==1):
  #     sel_class = row_0[1]
  #   else:
  #     sel_class = class_labels[max_cnt_class_num]

  #   ensemble_csv_wr.writerow([row_0[0], sel_class])

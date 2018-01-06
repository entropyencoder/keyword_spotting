import csv

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
with open("./test_ensemble_normal_CNN_up_0.50_DS_CNN3_up_0.60_GRU.csv", 'w') as ensemble_csv:
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
  # Ensemble 4
  pred_csv_0 = open("./test_normal_CNN_up_0.50.csv", "r")
  pred_csv_1 = open("./test_DS_CNN3_up_0.60.csv", "r")
  pred_csv_2 = open("./test_GRU_L.csv", "r")

  # Remove header
  pred_csv_0.readline()
  pred_csv_1.readline()
  pred_csv_2.readline()

  pred_csv_rd_0 = csv.reader(pred_csv_0)
  pred_csv_rd_1 = csv.reader(pred_csv_1)
  pred_csv_rd_2 = csv.reader(pred_csv_2)

  ensemble_csv_wr.writerow(['fname', 'label'])

  for row_0, row_1, row_2 in zip(pred_csv_rd_0, pred_csv_rd_1, pred_csv_rd_2):
    # Simple majority voting from three predictions
    # The highest priority is 'DSCNN'.
    if row_1[1]==row_2[1]:
      sel_class = row_1[1]
    else:
      sel_class = row_0[1]

    # # Post-process "unknown" classes
    # if sel_class=="unknown":
    #   if row_1[1]!="unknown":
    #     sel_class = row_1[1]
    #   elif row_2[1]!="unknown":
    #     sel_class = row_2[1]
    #   else:
    #     sel_class = row_0[1]

    ensemble_csv_wr.writerow([row_0[0], sel_class])


  # for row in dscnn_rd:
  #   print(row)
  #
  # for row in gru_rd:
  #   print(row)

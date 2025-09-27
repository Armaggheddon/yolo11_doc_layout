from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data="/home/user/doc_layout/doclaynet.yaml",
    epochs=30,
    imgsz=1280,
    batch=128,
    save=True,
    device=[0, 1, 2, 3],
    close_mosaic=20,
    patience=5,
)

# train1 results
# YOLO11n summary (fused): 100 layers, 2,584,297 parameters, 0 gradients, 6.3 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 203/203 5.8it/s 35.0s
#                    all       6489      99816       0.83      0.768       0.85      0.636
#                Caption       1067       1763      0.903      0.794      0.903      0.762
#               Footnote        179        312      0.703      0.487      0.595      0.418
#                Formula        548       1894       0.87      0.704      0.815      0.563
#              List-item       1687      13320      0.807      0.854      0.891      0.726
#            Page-footer       5134       5571      0.911      0.878      0.919      0.514
#            Page-header       3612       6683      0.885      0.645      0.869      0.515
#                Picture       1479       2775      0.793      0.851      0.901      0.804
#         Section-header       4506      15744      0.849      0.838      0.909      0.551
#                  Table       1478       2269      0.824      0.841      0.883      0.805
#                   Text       5762      49186      0.896      0.873      0.937      0.762
#                  Title        155        299      0.691      0.682      0.723      0.576

# train3
# YOLO11n summary (fused): 100 layers, 2,584,297 parameters, 0 gradients, 6.3 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 102/102 3.7it/s 27.7s
#                    all       6489      99816      0.871      0.819      0.892      0.694
#                Caption       1067       1763      0.902      0.842      0.922      0.805
#               Footnote        179        312       0.81       0.58      0.719      0.494
#                Formula        548       1894      0.871      0.778      0.856      0.617
#              List-item       1687      13320      0.886      0.844      0.911      0.759
#            Page-footer       5134       5571      0.858      0.932      0.948       0.62
#            Page-header       3612       6683      0.942      0.834      0.951      0.687
#                Picture       1479       2775       0.85      0.852       0.91      0.786
#         Section-header       4506      15744      0.909      0.869      0.941      0.659
#                  Table       1478       2269      0.824      0.825      0.872      0.686
#                   Text       5762      49186      0.896      0.893      0.944      0.791
#                  Title        155        299      0.831      0.758      0.839      0.728

# train2 is train3 last.pt with 10 more epochs (total 20)
# YOLO11n summary (fused): 100 layers, 2,584,297 parameters, 0 gradients, 6.3 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 102/102 3.7it/s 27.7s
#                    all       6489      99816      0.884       0.84      0.906      0.719
#                Caption       1067       1763      0.905      0.853      0.929      0.814
#               Footnote        179        312      0.852      0.664      0.776      0.552
#                Formula        548       1894      0.883       0.81      0.878      0.649
#              List-item       1687      13320      0.892      0.878      0.922      0.785
#            Page-footer       5134       5571       0.88      0.933      0.958      0.641
#            Page-header       3612       6683      0.936      0.859      0.958      0.714
#                Picture       1479       2775      0.852       0.85      0.907      0.792
#         Section-header       4506      15744      0.918      0.878      0.948      0.692
#                  Table       1478       2269      0.828      0.835      0.874      0.699
#                   Text       5762      49186       0.91      0.901      0.951      0.811
#                  Title        155        299      0.869      0.779      0.865      0.757

# train4 has same configuration as train3 but goes on for 40 epochs with batch 128
# YOLO11n summary (fused): 100 layers, 2,584,297 parameters, 0 gradients, 6.3 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 102/102 3.9it/s 26.3s
#                    all       6489      99816      0.898      0.847      0.917      0.716
#                Caption       1067       1763       0.93      0.862      0.945      0.838
#               Footnote        179        312      0.853      0.728      0.805      0.551
#                Formula        548       1894      0.879      0.796      0.881      0.667
#              List-item       1687      13320      0.909      0.891       0.93      0.785
#            Page-footer       5134       5571      0.911      0.938      0.968      0.629
#            Page-header       3612       6683      0.939      0.837      0.954       0.66
#                Picture       1479       2775      0.874      0.875       0.93       0.82
#         Section-header       4506      15744      0.912      0.882      0.948       0.64
#                  Table       1478       2269      0.851      0.835      0.893      0.716
#                   Text       5762      49186      0.918       0.91      0.955      0.798
#                  Title        155        299      0.904      0.769       0.88      0.777

# train5
# YOLO11s summary (fused): 100 layers, 9,417,057 parameters, 0 gradients, 21.3 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 102/102 2.5it/s 40.7s
#                    all       6489      99816      0.908      0.862      0.929      0.754
#                Caption       1067       1763      0.927      0.876      0.956       0.86
#               Footnote        179        312      0.878      0.715      0.827      0.586
#                Formula        548       1894      0.891      0.848      0.905      0.711
#              List-item       1687      13320      0.918      0.898      0.941      0.817
#            Page-footer       5134       5571      0.915      0.947      0.971      0.685
#            Page-header       3612       6683      0.938      0.859      0.959      0.716
#                Picture       1479       2775      0.881       0.88      0.938      0.834
#         Section-header       4506      15744      0.931      0.895      0.954      0.689
#                  Table       1478       2269      0.866      0.859      0.905      0.747
#                   Text       5762      49186      0.929      0.919      0.962      0.824
#                  Title        155        299      0.909      0.786      0.905      0.824

# train6
# YOLO11m summary (fused): 125 layers, 20,038,513 parameters, 0 gradients, 67.7 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 203/203 2.9it/s 1:11
#                    all       6489      99816      0.918      0.862      0.933      0.766
#                Caption       1067       1763      0.946      0.867      0.954      0.865
#               Footnote        179        312      0.936      0.746      0.862      0.592
#                Formula        548       1894      0.867      0.836      0.899       0.72
#              List-item       1687      13320      0.929        0.9      0.944      0.817
#            Page-footer       5134       5571      0.911      0.952      0.972      0.684
#            Page-header       3612       6683      0.955      0.843      0.958      0.768
#                Picture       1479       2775      0.913      0.873      0.938       0.84
#         Section-header       4506      15744      0.937      0.893      0.956      0.722
#                  Table       1478       2269      0.875       0.86      0.907      0.753
#                   Text       5762      49186      0.947      0.916      0.965      0.843
#                  Title        155        299      0.883      0.799      0.902      0.824

# train7 is yolo11n batch 256 with 40 epochs, no mosaic and patience=5, batch=256
# YOLO11n summary (fused): 100 layers, 2,584,297 parameters, 0 gradients, 6.3 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 51/51 1.9it/s 27.0s
#                    all       6489      99816      0.865      0.842      0.902      0.723
#                Caption       1067       1763      0.893      0.863      0.941       0.83
#               Footnote        179        312      0.833       0.67      0.756      0.513
#                Formula        548       1894      0.832       0.81      0.857      0.642
#              List-item       1687      13320      0.881      0.894      0.926      0.799
#            Page-footer       5134       5571      0.899       0.92      0.946      0.648
#            Page-header       3612       6683      0.914      0.833      0.949      0.752
#                Picture       1479       2775      0.834      0.866      0.925      0.809
#         Section-header       4506      15744       0.89       0.88      0.941      0.698
#                  Table       1478       2269      0.769       0.87      0.866      0.694
#                   Text       5762      49186      0.908      0.884      0.947      0.811
#                  Title        155        299      0.866      0.775      0.868      0.754

# train8 is batch 128 and mosaic for first 10 epochs only, trained 20 epochs
# YOLO11n summary (fused): 100 layers, 2,584,297 parameters, 0 gradients, 6.3 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 102/102 3.8it/s 27.1s
#                    all       6489      99816      0.889      0.836      0.906      0.715
#                Caption       1067       1763      0.913      0.852      0.936      0.825
#               Footnote        179        312      0.839       0.67      0.754      0.513
#                Formula        548       1894      0.866      0.799      0.874      0.651
#              List-item       1687      13320      0.901      0.878      0.925      0.791
#            Page-footer       5134       5571      0.887      0.932      0.959      0.638
#            Page-header       3612       6683      0.921      0.844      0.941      0.696
#                Picture       1479       2775       0.86      0.854      0.922      0.807
#         Section-header       4506      15744      0.913      0.876      0.944      0.672
#                  Table       1478       2269      0.856      0.829      0.889      0.708
#                   Text       5762      49186      0.916      0.901      0.953      0.804
#                  Title        155        299      0.905      0.761      0.868      0.757

# train9 is batch 128, close mosaic = 20, total 30 epochs (10 epochs w mosaic, 20 last without), patience=5
# YOLO11n summary (fused): 100 layers, 2,584,297 parameters, 0 gradients, 6.3 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 102/102 3.7it/s 27.7s
#                    all       6489      99816      0.889      0.852      0.915      0.716
#                Caption       1067       1763      0.927      0.873      0.946      0.838
#               Footnote        179        312      0.855      0.698      0.786       0.54
#                Formula        548       1894      0.876      0.833      0.886      0.669
#              List-item       1687      13320        0.9       0.89      0.929      0.787
#            Page-footer       5134       5571      0.915      0.933       0.96      0.636
#            Page-header       3612       6683      0.943      0.844      0.954      0.666
#                Picture       1479       2775      0.894      0.871       0.93      0.821
#         Section-header       4506      15744      0.899      0.888      0.949      0.637
#                  Table       1478       2269       0.83      0.845      0.885      0.714
#                   Text       5762      49186      0.916      0.908      0.955      0.798
#                  Title        155        299      0.828      0.793       0.88      0.773
print(results)
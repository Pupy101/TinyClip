# Homemade CLIP with training only image part.

## For use:
1. Download dataset; (for training I use 3 datasets [coco](https://www.kaggle.com/mrviswamitrakaushik/image-captioning-data), [flickr8k](https://www.kaggle.com/ashish2001/original-flickr8k-dataset), [flikr30k](https://www.kaggle.com/adityajn105/flickr30k))
2. Then i preprocess datasets with scrip to create .csv needed for training (in colab it):
```bash
!python /content/CLIP/utils/preprocessing_dataset.py \
    --dir_jsons '/content/caption_datasets' --coco_train '/content/train2014' \
    --coco_valid '/content/val2014' --flickr8k '/content/Flickr8k_Dataset/Flicker8k_Dataset' \
    --flickr30k '/content/Images/flickr30k_images' --target_csv '/content'
```
3. Configure the file `config.py`;
4. Start:
```bash 
!python main.py
```

#### If you don't need training skip step 1 and step 2.

---

#### Pretrained weights:
```coming soon```
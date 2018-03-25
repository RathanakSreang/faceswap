# faceswap
This is a faceswap application where we swap the Rock face to our face or our face to the Rock.

# Development
Create folder name `the_rock`
To download image file run `python download_image_from_url.py -u urls.txt -o the_rock`

To detect and crop `The Rock` face run `python save_face_from_pic.py -s the_rock -p the_rock`
To detect and crop your face from webcam run `python save_face_from_video.py -s your_name`

To train the model for `The Rock` run `python train.py -m the_rock -n the_rock`
To train the model for your face run `python train.py -m your_name -n your_name`

To test the model swap to `The Rock` run `python test_model.py -m the_rock`
To test the model swap to you run `python test_model.py -m your_name`

To learn more feel free to read source code.
# Deploy
None

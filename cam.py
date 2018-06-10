import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import os

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel,self).__init__()
        self.cnn1=nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.relu1=nn.ReLU()
        self.maxpool1=nn.MaxPool2d(kernel_size=2)
        self.cnn2=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2)
        self.relu2=nn.ReLU()
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        self.cnn3=nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2)
        self.relu3=nn.ReLU()
        self.maxpool3=nn.MaxPool2d(kernel_size=2)
        self.fc1=nn.Linear(10368,6)

    def forward(self,x):
        out=self.cnn1(x)
        out=self.relu1(out)
        out=self.maxpool1(out)
        out=self.cnn2(out)
        out=self.relu2(out)
        out=self.maxpool2(out)
        out=self.cnn3(out)
        out=self.relu3(out)
        out=self.maxpool3(out)
        out=out.view(out.size(0),-1)
        out=self.fc1(out)
        
        return out

test_model = CNNModel()
test_model.load_state_dict(torch.load('model_chk.pt'))

def main():
    emojis = get_emojis()
    cap = cv2.VideoCapture(0)
    x, y, w, h = 300, 50, 350, 350

    while (cap.isOpened()):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        pred = Classify(test_model, img)
        print(pred[0])
        img = overlay(img,emojis[pred[0]], 400, 250, 90, 90)
        x, y, w, h = 300, 50, 350, 350
        cv2.imshow("Frame", img)
        k = cv2.waitKey(10)
        if k == 27:
            break

def Classify(model, image):
    processed = process_image(image)
    outputs=test_model(processed)
    _,predicted=torch.max(outputs.data,1)
    return predicted

def process_image(img):
    image_x = 64
    image_y = 64
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float64)
    img = np.reshape(img, (image_x, image_y,-1))
    #print img.shape
    ptLoader = transforms.Compose([transforms.ToTensor()])
    img = ptLoader( img ).float()
    img = Variable( img.unsqueeze(0), volatile=True  )
    #print img.shape
    #print img
    return img

def get_emojis():
    emojis_folder = 'hand_symb/'
    emojis = []
    for emoji in range(len(os.listdir(emojis_folder))):
        #print(emoji)
        emojis.append(cv2.imread(emojis_folder+str(emoji)+'.png', -1))
    #print emojis
    return emojis

def overlay(image, emoji, x,y,w,h):
    emoji = cv2.resize(emoji, (w, h))
    try:
        image[y:y+h, x:x+w] = blend_transparent(image[y:y+h, x:x+w], emoji)
    except:
        pass
    return image

def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

Classify(test_model, np.zeros((64,64,3), dtype=np.uint8))
main()
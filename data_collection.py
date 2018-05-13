import pygame
import pygame.camera
from pygame.locals import *
import Image,os,sys

DEVICE = '/dev/video0'
SIZE = (640,480)
FILENAME = 'capture.jpg'
i=0

def camstream(path='./testimgs/'):
    pygame.init()
    pygame.camera.init()
    display = pygame.display.set_mode(SIZE, 0)
    camera = pygame.camera.Camera(DEVICE, SIZE)
    camera.start()
    screen = pygame.surface.Surface(SIZE, 0, display)
    capture = True
    global i
    while capture:
        screen = camera.get_image(screen)
        display.blit(screen, (0,0))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == QUIT:
                capture = False
            elif event.type == KEYDOWN and event.key == K_s:
                snapshot = pygame.transform.scale(screen,(300,300))
                pygame.image.save(snapshot, path+str(i)+'.png')
                i+=1
                print 'captured this frame.........'
    camera.stop()
    pygame.quit()
    return

if __name__ == '__main__':
    

    path='./data/ONE_'
    print 'streaming........'
    camstream(path)
    print 'finished placing testing images of size 256x256.......'
    print 'completed....'
'''
	else:
		if(direction== '0'):
			os.system('python main.py --phase=test --checkpoint_dir=. --dataset_dir_path=datasets/ --dataset_dir=neutral_surprised --which_direction=AtoB')
		else:
			os.system('python main.py --phase=test --checkpoint_dir=. --dataset_dir_path=datasets/ --dataset_dir=neutral_surprised --which_direction=BtoA')
'''
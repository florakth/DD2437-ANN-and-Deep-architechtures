from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet
import matplotlib.pyplot as plt

if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    ''' restricted boltzmann machine '''
       
    print ("\nStarting a Restricted Boltzmann Machine..")
    total_err = []
    rbm500 = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=500,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20
    )
    #print(train_imgs.shape)
    
    total_err.append(rbm500.cd1(visible_trainset=train_imgs, n_iterations=10))
   
     rbm400 = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=400,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20
    )
    #print(train_imgs.shape)
    
    total_err.append(rbm400.cd1(visible_trainset=train_imgs, n_iterations=10))

    rbm300 = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=300,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20
    )
    #print(train_imgs.shape)
    
    total_err.append(rbm500.cd1(visible_trainset=train_imgs, n_iterations=10))
    

     rbm200 = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=200,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20
    )
    #print(train_imgs.shape)
    
    total_err.append(rbm200.cd1(visible_trainset=train_imgs, n_iterations=10))


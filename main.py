# -*- coding: utf-8 -*-

import engine
from torch.utils.data import DataLoader
import torch

# GET COMMANDLINE ARGS
args = engine.get_args()

# LOADING MODEL
model = engine.load_model(args.model,args.noc)

# DATAPATHS
train_images = args.dataset_path+'/train/images'
train_labels = args.dataset_path+'/train/labels'
val_images = args.dataset_path+'/validation/images'
val_labels = args.dataset_path+'/validation/labels'
test_images = args.dataset_path+'/test/images'
test_labels = args.dataset_path+'/test/labels'

# DATA LOADERS
train_loader = DataLoader(engine.getDataset(train_images,
                                              train_labels,
                                              size = (360,480)),
                          batch_size=args.batch_size,
                          num_workers=args.num_of_workers,
                          shuffle=True)

val_loader = DataLoader(engine.getDataset(val_images,
                                            val_labels,
                                            size = (360,480)),
                        batch_size=args.batch_size,
                        num_workers=args.num_of_workers,
                        shuffle=False)
test_loader = DataLoader(engine.getDataset(test_images,
                                            test_labels,
                                            size = (360,480)),
                        batch_size=args.batch_size,
                        num_workers=args.num_of_workers,
                        shuffle=False)

# TRAINING

if args.fresh_train:
    trainer = engine.Trainer(model.cuda(),train_loader,val_loader,args.save_path,args.max_epochs,args.noc)
    trained_model = trainer.train()
else:
    trained_model = model.load_state_dict(torch.load(args.save_path+'/best_model.pth')).cuda()

# TESTING
tester_test = engine.Tester(trained_model,test_loader,args.save_path+'/eval_test',args.noc)
tester_test.test()
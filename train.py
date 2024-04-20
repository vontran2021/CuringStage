import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets

from tqdm import tqdm


from model import efficientnetv2_s as create_model




def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    if os.path.exists("./New_weights/S/F30_F") is False:
        os.makedirs("./New_weights/S/F30_F")

    img_size = {"s": [380, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                   transforms.CenterCrop(img_size[num_model][1]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path =  "/test/TobaccoStage/data_set/F30"  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

   
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=11)
    with open('TMBclass_Flue_TobaccoStage.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 64
    epochs = 200
    num_saved_weights = 5
    nw = 8  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # create model
    net = create_model(num_classes=11)
    # load pretrain weights

    
    model_weight_path = "./New_weights/S/F/checkpoint_F_200.pth"
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location='cpu')
    # #
    # # delete classifier weights
    # pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    # missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)
    missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)
    # #
    # # # freeze features weights
    # # for param in net.features.parameters():
    # #     param.requires_grad = False

    net.to(device)
    doc1 = open('./New_weights/S/F30_F/train_aac.txt', 'w')
    doc2 = open('./New_weights/S/F30_F/val_acc.txt', 'w')
    doc3 = open('./New_weights/S/F30_F/train_loss.txt', 'w')
    doc4 = open('./New_weights/S/F30_F/val_Loss.txt', 'w')
    doc5 = open('./New_weights/S/F30_F/time.txt', 'w')
    doc6 = open('./New_weights/S/F30_F/lr.txt', 'w')
    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    # params = [p for p in net.parameters() if p.requires_grad]
    pg = [p for p in net.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=0.0001, momentum=0.9, weight_decay=5E-5)
    optimizer = optim.Adam(pg, lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=120, eta_min=1e-6)

    best_acc = 0.0
    
    save_path = './New_weights/S/F30_F'
    
    train_steps = len(train_loader)
    t1 = time.time()
    for epoch in range(epochs):
        # train
        accc = 0.0
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            # scheduler.step()
            predict_x = torch.max(logits, dim=1)[1]
            accc += (predict_x == labels.to(device)).sum().item()
            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
            print(loss, file=doc3)
        train_acc = accc / train_num
        print('train_accuarcy:{:.4f}'.format(train_acc))
        print(train_acc, file=doc1)
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]

                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
        val_accurate = acc / val_num
        print(val_accurate, file=doc2)
        eval_loss = running_loss / train_steps
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, eval_loss, val_accurate))
        print(eval_loss, file=doc4)
        print('train_time:{:.4f}'.format(time.perf_counter() - t1))
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        print(optimizer.state_dict()['param_groups'][0]['lr'], file=doc6)

        
        new_weight_filename = f"checkpoint_F30_{epoch + 1}.pth"
        new_weight_path = os.path.join(save_path, new_weight_filename)
        
        torch.save(net.state_dict(), new_weight_path)
       
        existing_weights = sorted([file for file in os.listdir(save_path) if file.endswith(".pth")],
                                 key=lambda x:os.path.getmtime(os.path.join(save_path, x)),
                                  reverse=True)
        
        if val_accurate > best_acc:
            best_acc = val_accurate
      
       
        if len(existing_weights) > num_saved_weights:
            os.remove(os.path.join(save_path, existing_weights[-1]))

        print('MaxAccuracy:{:.4f}'.format(best_acc))
      


    print('train_time:{:.4f}'.format(time.perf_counter() - t1))
    doc1.close()
    doc2.close()
    doc3.close()
    doc4.close()
    doc6.close()
    time_end = time.time() 
    time_sum = time_end - time_start  
    print(time_sum)
    print(time_sum, file=doc5)
    doc5.close()
    print('Finished Training')


if __name__ == '__main__':
    time_start = time.time()  
    main()

import os
import torch
import torch.nn as nn

from torch.autograd import Variable

def get_outputs(model, instances, flow=False, use_cuda=False):

    if flow:
        frames = Variable(instances["frames"])
        flow_x = Variable(instances["flow_x"])
        flow_y = Variable(instances["flow_y"])

        if use_cuda:
            frames = frames.cuda()
            flow_x = flow_x.cuda()
            flow_y = flow_y.cuda()

        outputs = model(frames, flow_x, flow_y)

    else:
        instances = Variable(instances)
        if use_cuda:
            instances = instances.cuda()

        outputs = model(instances)

    return outputs

def evaluate(model, dataloader, flow=False, use_cuda=False):
    loss = 0
    correct = 0
    total = 0

    # Switch to evaluation mode.
    model.eval()

    for i, samples in enumerate(dataloader):
        outputs = get_outputs(model, samples["instance"], flow=flow,
                              use_cuda=use_cuda)
        
        labels = Variable(samples["label"])
        if use_cuda:
            labels = labels.cuda()

        loss += nn.CrossEntropyLoss(size_average=False)(outputs, labels).data[0]

        score, predicted = torch.max(outputs, 1)
        correct += (labels.data == predicted.data).sum()
        
        total += labels.size(0)

    acc = correct / total
    loss /= total

    return loss, acc

def train(model, num_epochs, train_set, dev_set, lr=1e-3, batch_size=32,
          start_epoch=1, log=10, checkpoint_path=None, validate=True,
          resume=False, flow=False, use_cuda=False):

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True)

    # Must be sequential b/c this is used for evaluation.
    train_loader_sequential = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=False)
    dev_loader = torch.utils.data.DataLoader(
        dataset=dev_set, batch_size=batch_size, shuffle=False)

    # Use Adam optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Record loss + accuracy.
    hist = []

    # Check if we are resuming training from a previous checkpoint.
    if resume:
        checkpoint = torch.load(os.path.join(
            checkpoint_path, "model_epoch%d.chkpt" % (start_epoch - 1)))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        hist = checkpoint["hist"]

    if use_cuda:
        model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Switch to train mode.
        model.train()

        for i, samples in enumerate(train_loader):

            labels = Variable(samples["label"])
            if use_cuda:
                labels = labels.cuda()

            # Zero out gradient from previous iteration.
            optimizer.zero_grad()

            # Forward, backward, and optimize.
            outputs = get_outputs(model, samples["instance"], flow=flow,
                                  use_cuda=use_cuda)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % log == 0:
                print("epoch %d/%d, iteration %d/%d, loss: %s"
                      % (epoch, start_epoch + num_epochs - 1, i + 1,
                      len(train_set) // batch_size, loss.data[0]))
        
        # Get overall loss & accuracy on training set.
        train_loss, train_acc = evaluate(model, train_loader_sequential,
                                         flow=flow, use_cuda=use_cuda)

        if validate:
            # Get overall loss & accuracy on dev set.
            dev_loss, dev_acc = evaluate(model, dev_loader, flow=flow,
                                         use_cuda=use_cuda)

            print("epoch %d/%d, train_loss = %s, traic_acc = %s, "
                  "dev_loss = %s, dev_acc = %s"
                  % (epoch, start_epoch + num_epochs - 1,
                  train_loss, train_acc, dev_loss, dev_acc))

            hist.append({
                "train_loss": train_loss, "train_acc": train_acc,
                "dev_loss": dev_loss, "dev_acc": dev_acc
            })
        else:
            print("epoch %d/%d, train_loss = %s, train_acc = %s" % (epoch,
                  start_epoch + num_epochs - 1, train_loss, train_acc))

            hist.append({
                "train_loss": train_loss, "train_acc": train_acc
            })

        optimizer.zero_grad()
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "hist": hist
        }

        # Save checkpoint.
        torch.save(checkpoint, os.path.join(
            checkpoint_path, "model_epoch%d.chkpt" % epoch))


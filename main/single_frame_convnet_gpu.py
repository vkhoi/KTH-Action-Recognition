from dataset import *
from torch.autograd import Variable

import argparse
import torch
import torch.nn as nn

class SingleFrameConvNet(nn.Module):
	def __init__(self):
		super(SingleFrameConvNet, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2))

		self.conv2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2))

		self.conv3 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2))

		self.fc1 = nn.Linear(17920, 256)
		self.fc2 = nn.Linear(256, 6)

	def forward(self, x):
		out = self.conv1(x)
		out = nn.Dropout(0.5)(out)
		out = self.conv2(out)
		out = nn.Dropout(0.5)(out)
		out = self.conv3(out)
		out = nn.Dropout(0.5)(out)

		out = out.view(out.size(0), -1)
		out = self.fc1(out)
		out = nn.ReLU()(out)
		out = nn.Dropout(0.5)(out)
		out = self.fc2(out)

		return out

def evaluate(dataloader):
	loss = 0
	correct = 0
	total = 0

	for i, samples in enumerate(dataloader):
		images = Variable(samples["image"]).cuda()
		labels = Variable(samples["label"]).cuda()

		outputs = net(images)
		loss += (nn.CrossEntropyLoss(size_average=False)(outputs, labels)).data[0]

		score, predicted = torch.max(outputs, 1)
		correct += (labels.data == predicted.data).sum()
		
		total += labels.size(0)
		
	acc = correct / total
	loss /= total

	return loss, acc

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Single Frame ConvNet")
	parser.add_argument("--dataset_dir", type=str, default="data",
		help="directory to dataset")
	parser.add_argument("--batch_size", type=int, default=64,
		help="batch size for training (default: 64)")
	parser.add_argument("--epochs", type=int, default=3,
		help="number of epochs to train (default: 3)")
	parser.add_argument("--start_epoch", type=int, default=1,
		help="start index of epoch (default: 1)")
	parser.add_argument("--lr", type=int, default=0.001,
		help="learning rate for training (default: 0.001)")
	parser.add_argument("--log", type=int, default=10,
		help="how many batches to wait before outputing the training logs (default: 10)")
	parser.add_argument("--val", type=int, default=0,
		help="whether to run validation (default: 0)")
	args = parser.parse_args()

	print("Parsing arguments")
	dataset_dir = args.dataset_dir
	batch_size = args.batch_size
	num_epochs = args.epochs
	start_epoch = args.start_epoch
	learning_rate = args.lr
	log_interval = args.log
	flag_validate = args.val

	print("Loading training set")
	train_dataset = RawDataset(dataset_dir, "train")
	print("Done loading training set")

	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
		batch_size=batch_size, shuffle=True)
	train_loader_sequential = torch.utils.data.DataLoader(dataset=train_dataset,
		batch_size=batch_size, shuffle=False)

	if flag_validate:
		print("Loading dev set")
		dev_dataset = RawDataset(dataset_dir, "dev")
		print("Done loading dev set")
		dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset,
			batch_size=batch_size, shuffle=False)

	net = SingleFrameConvNet()
	optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
	hist = []

	if start_epoch > 1:
		print("Loading checkpoint %d" % (start_epoch - 1))
		checkpoint = torch.load(os.path.join(dataset_dir,
			"cnn_single_frame_epoch%d.chkpt" % (start_epoch - 1)))
		net.load_state_dict(checkpoint["model"])
		optimizer.load_state_dict(checkpoint["optimizer"])
		hist = checkpoint["hist"]

	net.cuda()

	criterion = nn.CrossEntropyLoss().cuda()  

	print("start training")
	for epoch in range(start_epoch, start_epoch + num_epochs):
		for i, samples in enumerate(train_loader):
			images = Variable(samples["image"]).cuda()
			labels = Variable(samples["label"]).cuda()

			optimizer.zero_grad()
			outputs = net(images)

			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			if (i+1) % log_interval == 0:
				print("epoch %d/%d, iteration %d/%d, Loss: %s" % (epoch,
					start_epoch + num_epochs - 1, i + 1,
					len(train_dataset) // batch_size, loss.data[0]))
		
		print("Evaluating...")
		train_loss, train_acc = evaluate(train_loader_sequential)
		if flag_validate:
			dev_loss, dev_acc = evaluate(dev_loader)
			print("epoch %d/%d, train_loss = %s, traic_acc = %s, "
				"dev_loss = %s, dev_acc = %s"
				% (epoch, start_epoch + num_epochs - 1, train_loss, train_acc,
					dev_loss, dev_acc))

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
			"model": net.state_dict(),
			"optimizer": optimizer.state_dict(),
			"hist": hist
		}
		torch.save(checkpoint,
			os.path.join(dataset_dir, "cnn_single_frame_epoch%d.chkpt" % epoch))


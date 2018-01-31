import argparse
import pickle

from dataset import *
from models.cnn_block_frame import CNNBlockFrame
from torch.autograd import Variable

CATEGORIES = [
    "boxing",
    "handclapping",
    "handwaving",
    "jogging",
    "running", 
    "walking"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default="data",
        help="directory to dataset")
    parser.add_argument("--model_dir", type=str, default="",
        help="directory to model")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    model_dir = args.model_dir

    print("Loading dataset")
    train_dataset = BlockFrameDataset(dataset_dir, "train")
    videos = pickle.load(open(os.path.join(dataset_dir, "test.p"), "rb"))

    print("Loading model")
    chkpt = torch.load(model_dir, map_location=lambda storage, loc: storage)
    model = CNNBlockFrame()
    model.load_state_dict(chkpt["model"])

    # Number of correct classified videos.
    correct = 0

    model.eval()
    for i in range(len(videos)):
        video = videos[i]

        # Class probabilities.
        P = np.zeros(6, dtype=np.float32)

        current_block = []
        cnt = 0

        for frame in video["frames"]:
            current_block.append(frame)
            if len(current_block) % 15 == 0:
                cnt += 1
                current_block = np.array(
                    current_block, dtype=np.float32).reshape((1, 15, 60, 80))
                current_block_shifted = current_block - train_dataset.mean
                tensor = torch.from_numpy(current_block_shifted)
                instance = Variable(tensor.unsqueeze(0))
                score = model(instance).data[0].numpy()

                score -= np.max(score)
                p = np.e**score / np.sum(np.e**score)
                P += p

                current_block = []

        P /= cnt
        pred = CATEGORIES[np.argmax(P)]
        if pred == video["category"]:
            correct += 1

        if i > 0 and i % 10 == 0:
            print("Done %d/%d videos" % (i, len(videos)))

    print("%d/%d correct" % (correct, len(videos)))
    print("Accuracy: %.9f" % (correct / len(videos)))


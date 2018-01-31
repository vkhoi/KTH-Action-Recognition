import argparse
import pickle

from dataset import *
from models.cnn_block_frame_flow import CNNBlockFrameFlow
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
    train_dataset = BlockFrameFlowDataset(dataset_dir, "train")
    video_frames = pickle.load(open("data/test.p", "rb"))
    video_flows = pickle.load(open("data/test_flow.p", "rb"))

    print("Loading model")
    chkpt = torch.load(model_dir, map_location=lambda storage, loc: storage)
    model = CNNBlockFrameFlow()
    model.load_state_dict(chkpt["model"])

    # Number of correct classified videos.
    correct = 0

    model.eval()
    for i in range(len(video_frames)):
        frames = video_frames[i]["frames"]
        flow_x = [0] + video_flows[i]["flow_x"]
        flow_y = [0] + video_flows[i]["flow_y"]

        # Class probabilities.
        P = np.zeros(6, dtype=np.float32)

        current_block_frame = []
        current_block_flow_x = []
        current_block_flow_y = []
        cnt = 0

        for i_frame in range(len(frames)):
            current_block_frame.append(frames[i_frame])

            if i_frame % 15 > 0:
                current_block_flow_x.append(flow_x[i_frame])
                current_block_flow_y.append(flow_y[i_frame])

            if (i_frame + 1) % 15 == 0:
                cnt += 1

                current_block_frame = np.array(
                    current_block_frame,
                    dtype=np.float32).reshape((1, 15, 60, 80))

                current_block_flow_x = np.array(
                    current_block_flow_x,
                    dtype=np.float32).reshape((1, 14, 30, 40))

                current_block_flow_y = np.array(
                    current_block_flow_y,
                    dtype=np.float32).reshape((1, 14, 30, 40))

                current_block_frame -= train_dataset.mean["frames"]
                current_block_flow_x -= train_dataset.mean["flow_x"]
                current_block_flow_y -= train_dataset.mean["flow_y"]

                tensor_frames = torch.from_numpy(current_block_frame)
                tensor_flow_x = torch.from_numpy(current_block_flow_x)
                tensor_flow_y = torch.from_numpy(current_block_flow_y)
                
                instance_frames = Variable(tensor_frames.unsqueeze(0))
                instance_flow_x = Variable(tensor_flow_x.unsqueeze(0))
                instance_flow_y = Variable(tensor_flow_y.unsqueeze(0))

                score = model(instance_frames, instance_flow_x,
                              instance_flow_y).data[0].numpy()

                score -= np.max(score)
                p = np.e**score / np.sum(np.e**score)
                P += p

                current_block_frame = []
                current_block_flow_x = []
                current_block_flow_y = []

        P /= cnt
        pred = CATEGORIES[np.argmax(P)]
        if pred == video_frames[i]["category"]:
            correct += 1

        if i > 0 and i % 10 == 0:
            print("Done %d/%d videos" % (i, len(video_frames)))

    print("%d/%d correct" % (correct, len(video_frames)))
    print("Accuracy: %.9f" % (correct / len(video_frames)))


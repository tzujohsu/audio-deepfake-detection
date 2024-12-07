import csv


def convert_txt_to_csv(path_txt: str, path_csv: str):
    """Convert ASVSpoof2019 original protocol file (txt format) to csv format.

    Args:
        path_txt (str): Path to ASVSpoof2019 protocol (.txt)
        path_csv (str): Path for converted csv file
    """
    with open(path_txt) as rf:
        with open(path_csv, "w") as wf:

            readfile = rf.readlines()
            writer = csv.writer(wf, delimiter=",")

            # Write header
            cols = ["speaker_id", "utt_id", "config", "attacks", "key"]
            writer.writerow(cols)
            for read_text in readfile:

                read_text = read_text.split()
                writer = csv.writer(wf, delimiter=",")
                writer.writerow(read_text)


if __name__ == "__main__":
    # Set your own path to `path_txt`
    convert_txt_to_csv("/home/tzujohsu/audio-deepfake/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019_aug.LA.cm.train.trn.txt", "./aug_protocol.csv")

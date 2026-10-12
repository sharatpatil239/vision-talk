import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


def xml_to_csv(path):
    xml_list = []

    for xml_file in os.listdir(path):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(path, xml_file)

        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for member in root.findall("object"):
            value = (
                root.find("filename").text,
                int(root.find("size/width").text),
                int(root.find("size/height").text),
                member[0].text,
                int(member.find("bndbox/xmin").text),
                int(member.find("bndbox/ymin").text),
                int(member.find("bndbox/xmax").text),
                int(member.find("bndbox/ymax").text),
            )

            xml_list.append(value)

    column_name = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]

    return pd.DataFrame(xml_list, columns=column_name)


def class_text_to_int(row_label, label_map_dict):
    return label_map_dict[row_label]


def create_tf_example(group, filename, path, label_map_dict):

    image_path = os.path.join(path, filename)

    with tf.io.gfile.GFile(image_path, "rb") as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)

    image = Image.open(encoded_jpg_io)

    width, height = image.size

    filename_bytes = filename.encode("utf8")

    image_format = b"jpg"

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for _, row in group.iterrows():

        xmins.append(row["xmin"] / width)
        xmaxs.append(row["xmax"] / width)

        ymins.append(row["ymin"] / height)
        ymaxs.append(row["ymax"] / height)

        classes_text.append(row["class"].encode("utf8"))

        classes.append(
            class_text_to_int(row["class"], label_map_dict)
        )

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": dataset_util.int64_feature(height),
                "image/width": dataset_util.int64_feature(width),
                "image/filename": dataset_util.bytes_feature(filename_bytes),
                "image/source_id": dataset_util.bytes_feature(filename_bytes),
                "image/encoded": dataset_util.bytes_feature(encoded_jpg),
                "image/format": dataset_util.bytes_feature(image_format),
                "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
                "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
                "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
                "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
                "image/object/class/text": dataset_util.bytes_list_feature(classes_text),
                "image/object/class/label": dataset_util.int64_list_feature(classes),
            }
        )
    )

    return tf_example


def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-x",
        "--xml_dir",
        help="Path to XML files",
        required=True
    )

    parser.add_argument(
        "-l",
        "--labels_path",
        help="Path to label map",
        required=True
    )

    parser.add_argument(
        "-o",
        "--output_path",
        help="Path to output TFRecord",
        required=True
    )

    args = parser.parse_args()

    label_map = label_map_util.load_labelmap(args.labels_path)

    label_map_dict = label_map_util.get_label_map_dict(label_map)

    writer = tf.io.TFRecordWriter(args.output_path)

    examples = xml_to_csv(args.xml_dir)

    grouped = examples.groupby("filename")

    for group_name, group in grouped:

        tf_example = create_tf_example(
            group=group,filename=group_name,path=args.xml_dir,label_map_dict=label_map_dict
        )

        writer.write(tf_example.SerializeToString())

    writer.close()

    print(
        "Successfully created TFRecord:",
        args.output_path
    )


if __name__ == "__main__":
    main()
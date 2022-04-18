# create by andy at 2022/3/28
# reference: https://www.tensorflow.org/datasets/add_dataset

"""obt_dataset dataset."""
import glob
import os

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

# TODO(obt_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
This is the dataset that Andy needs to use for his graduation project.
There are about 90 hyperspectral image which dimension is (500 * 500 * 32) and their labels' are (500 * 500)

There are five value 1, 2, 3, 4, 255. And the corresponding semantics are vegetarian, building, bare_field, water and 
background.
"""

# TODO(obt_dataset): BibTeX citation
_CITATION = """\
@InProceedings{obt,
  author       = "Andy Z Wright",
  title        = "OBT dataset",
  booktitle    = "Classification based on morphology for hyperspectral imagery",
  year         = "2022",
  

}
"""


class ObtDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for obt_dataset dataset."""

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Please access me to get the data. Place the `obt_dataset.zip`
    file in the `~/tensorflow_datasets/downloads/manual/`.
    """

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(obt_dataset): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Tensor(shape=(500, 500, 32), dtype=tf.uint16),
                'segmentation_mask': tfds.features.Image(shape=(500, 500, 1), dtype=tf.uint8)
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'segmentation_mask'),  # Set to `None` to disable
            homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(obt_dataset): Downloads the data and defines the splits
        archive_path = dl_manager.manual_dir + "/obt_dataset.zip"

        extracted_path = dl_manager.extract(archive_path)

        # TODO(obt_dataset): Returns the Dict[split names, Iterator[Key, Example]]
        train_split = tfds.core.SplitGenerator(
            name="train",
            gen_kwargs={
                "path": extracted_path
            }
        )
        return [train_split]

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(obt_dataset): Yields (key, example) tuples from the dataset
        for f in glob.glob(f'{path}/*.npy'):
            dict_load = np.load(f, allow_pickle=True)
            yield os.path.basename(f), {
                'image': dict_load.item().get("image"),
                'segmentation_mask': dict_load.item().get("segmentation_mask"),
            }


if __name__ == '__main__':
    dataset = tfds.load("obt_dataset", with_info=False)
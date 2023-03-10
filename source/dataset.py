import config
from torchvision import datasets, transforms


class LFW(datasets.LFWPeople):
    """
    People dataset.
    """

    def __getitem__(self, index: int):
        """
        Get item from dataset;
        :param index: index;
        :return: triplet -- (augmented image, augmented image, label).
        """
        img = self._loader(self.data[index])
        target = self.targets[index]

        if self.transform is not None:
            img_1 = self.transform(img)
            img_2 = self.transform(img)
        else:
            img_1, img_2 = img, img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_1, img_2, target


augmentation = {
    "train": transforms.Compose(
        [
            transforms.RandomCrop(
                size=config.ORIGINAL_SIZE - config.ORIGINAL_SIZE // 4
            ),
            transforms.Resize(size=config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=1),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.05),
            transforms.ToTensor(),
        ]
    ),
    "valid": transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    ),
}

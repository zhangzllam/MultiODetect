import torchvision.transforms as transforms

def argumentation(size):
        transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=True),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform

'''def argumentation(size):
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=True),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        return transform'''
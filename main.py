from dataloaders import wikiart

training_dataset, validation_dataset = wikiart.get_dataset()

print(training_dataset)
print(validation_dataset)
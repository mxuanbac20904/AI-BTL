import csv
import random
import math

# Load data tu CSV file
def load_data(filename):
    with open(filename, mode='r', encoding='utf-8') as file:
        lines = csv.reader(file)
        dataset = list(lines)
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
    return dataset


# Phan chia tap du lieu theo class
def separate_data(dataset):
    separated = {}
    for vector in dataset:
        if vector[-1] not in separated:
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

# Phan chia tap du lieu thanh training va testing.
def split_data(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


# tinh toan gia tri trung binh cua moi thuoc tinh
def mean(numbers):
    return sum(numbers) / len(numbers)

# Tinh toan do lech chuan cho tung thuoc tinh
def standard_deviation(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / (len(numbers) - 1)
    return math.sqrt(variance)

# Gia tri trung binh , do lech chuan cho tap du lieu
def summarize(dataset):
    summaries = [(mean(attribute), standard_deviation(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]  # Remove the label column
    return summaries

def summarize_by_class(dataset):
    separated = separate_data(dataset)
    summaries = {}
    for classValue, instances in separated.items(): 
        summaries[classValue] = summarize(instances)
    return summaries

# Tinh toan xac suat theo phan phoi Gause cua bien lien tuc
def calculate_prob(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

# Tinh xac suat cho moi thuoc tinh phan chia theo class
def calculate_class_prob(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculate_prob(x, mean, stdev)
    return probabilities

# Du doan vector thuoc phan lop nao
def predict(summaries, inputVector):
    probabilities = calculate_class_prob(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

# Du doan tap du lieu testing thuoc vao phan lop nao
def get_predictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

# Tinh toan do chinh xac cua phan lop
def get_accuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / len(testSet)) * 100.0

def main():
    filename = 'tieu_duong.csv'
    splitRatio = 0.8
    dataset = load_data(filename)
    trainingSet, testSet = split_data(dataset, splitRatio)

    print('Data size: {0} \nTraining Size: {1} \nTest Size: {2}'.format(len(dataset), len(trainingSet), len(testSet)))
    # Prepare model
    summaries = summarize_by_class(trainingSet)
    # Test model
    predictions = get_predictions(summaries, testSet)
    accuracy = get_accuracy(testSet, predictions)
    print('Accuracy: {0}%'.format(round(accuracy, 4)))
    f = open("./test.csv", "w")
    f.write(input("Nhap gia tri: "))
    f.close()
    datatest = load_data('./test.csv')
    diagnose = get_predictions(summaries, datatest)
    print("Class:", diagnose)
    if diagnose == [1.0]:
        print("Co mac benh tieu duong.")
    else:
        print("Khong mac benh tieu duong.")
        
if __name__ == "__main__":
    main()
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <thread>
#include "yardimci.hpp"


// Veri noktas�n� temsil eden s�n�f
class DataPoint {
public:
	std::vector<double> features; // �zellik vekt�r�
	int label; // Etiket

	DataPoint(const std::vector<double>& features, int label) : features(features), label(label) {}
};


// �ki nokta aras�ndaki Euclidean uzakl���n� hesaplayan fonksiyon
double euclideanDistance(const std::vector<double>& point1, const std::vector<double>& point2) {
	double distance = 0.0;
	for (size_t i = 0; i < point1.size(); i++) {
		distance += pow(point1[i] - point2[i], 2);
	}
	return sqrt(distance);
}
// Paralel i� par�ac��� i�levi
void parallelDistanceCalculation(DataPoint& dataPoint, const std::vector<double>& unknown) {
	double distance = euclideanDistance(dataPoint.features, unknown);

	dataPoint.features = { distance }; // Uzakl��� �zellik vekt�r�ne kaydet
}


int kNNClassification(const std::vector<DataPoint>& trainingSet, const std::vector<double>& unknown, int k) {
	std::vector<std::pair<double, int>> distances; // Uzakl�k ve etiket �iftlerini i�eren vekt�r

	for (const auto& dataPoint : trainingSet) {
		double distance = euclideanDistance(dataPoint.features, unknown);
		distances.emplace_back(distance, dataPoint.label);
	}

	// Uzakl�klar� s�rala
	std::sort(distances.begin(), distances.end());

	// En yak�n k noktan�n s�n�flar�n� say
	std::vector<int> classCounts(k, 0);
	for (int i = 0; i < k; i++) {
		classCounts[distances[i].second]++;
	}

	// En fazla s�n�fa sahip olan s�n�f� bul
	int maxCount = 0;
	int predictedLabel = -1;
	for (size_t i = 0; i < classCounts.size(); i++) {
		if (classCounts[i] > maxCount) {
			maxCount = classCounts[i];
			predictedLabel = i;
		}
	}

	return predictedLabel;
}

// k-NN s�n�fland�rma i�lemini ger�ekle�tiren fonksiyon
int kNNClassification_thread(const std::vector<DataPoint>& trainingSet, const std::vector<double>& unknown, int k) {
	std::vector<DataPoint> distances = trainingSet;

	std::vector<std::thread> threads;
	for (auto& dataPoint : distances) {
		threads.emplace_back(parallelDistanceCalculation, std::ref(dataPoint), std::cref(unknown));
	}

	for (auto& thread : threads) {
		thread.join();
	}

	// Uzakl�klar� s�rala
	std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
		return a.features[0] < b.features[0];
		});

	// En yak�n k noktan�n s�n�flar�n� say
	std::vector<int> classCounts(k, 0);
	for (int i = 0; i < k; i++) {
		classCounts[distances[i].label]++;
	}

	// En fazla s�n�fa sahip olan s�n�f� bul
	int maxCount = 0;
	int predictedLabel = -1;
	for (size_t i = 0; i < classCounts.size(); i++) {
		if (classCounts[i] > maxCount) {
			maxCount = classCounts[i];
			predictedLabel = i;
		}
	}

	return predictedLabel;
}
// k-NN s�n�fland�rma i�lemini ger�ekle�tiren fonksiyon
int kNNClassification_openmp(const std::vector<DataPoint>& trainingSet, const std::vector<double>& unknown, int k) {
	std::vector<std::pair<double, int>> distances; // Uzakl�k ve etiket �iftlerini i�eren vekt�r
	int size = trainingSet.size();
#pragma omp parallel for
	for (int p = 0; p < size; p++) {
		double distance = euclideanDistance(trainingSet[p].features, unknown);

#pragma omp critical
		distances.emplace_back(distance, trainingSet[p].label);
	}

	// Uzakl�klar� s�rala
	std::sort(distances.begin(), distances.end());

	// En yak�n k noktan�n s�n�flar�n� say
	std::vector<int> classCounts(k, 0);
	for (int i = 0; i < k; i++) {
		classCounts[distances[i].second]++;
	}

	// En fazla s�n�fa sahip olan s�n�f� bul
	int maxCount = 0;
	int predictedLabel = -1;
	for (size_t i = 0; i < classCounts.size(); i++) {
		if (classCounts[i] > maxCount) {
			maxCount = classCounts[i];
			predictedLabel = i;
		}
	}

	return predictedLabel;
}

std::vector<DataPoint> generateRandomPoint(int trainingSetSize) {
	// Rastgele say� �reteci olu�tur


	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(0.0, 10.0);


	std::vector<DataPoint> trainingSet;


	for (int i = 0; i < trainingSetSize; i++) {
		double x = dist(gen);
		double y = dist(gen);
		int label = i % 2; // S�n�f etiketlerini s�rayla de�i�tir

		trainingSet.push_back(DataPoint({ x, y }, label));
	}


	return trainingSet;
}
int main() {

	int numPoint = 5000000; // point say�s�
	std::vector<DataPoint> trainingSet = generateRandomPoint(numPoint);



	// Bilinmeyen nokta
	std::vector<double> unknownPoint = { 5.0, 5.0 };

	// k-NN s�n�fland�rmas�n� ger�ekle�tir
	int k = 3;
	int predictedLabel;


	TIMERSTART(kNNClassification)
		predictedLabel = kNNClassification(trainingSet, unknownPoint, k);
	std::cout << "Bilinmeyen nokta etiketi (Seri): " << predictedLabel << std::endl;
	TIMERSTOP(kNNClassification)



	//	TIMERSTART(kNNClassification_thread)
	//	predictedLabel = kNNClassification_thread(trainingSet, unknownPoint, k);
	//std::cout << "Bilinmeyen nokta etiketi (PThread): " << predictedLabel << std::endl;
	//TIMERSTOP(kNNClassification_thread)



		TIMERSTART(kNNClassification_openmp)
		predictedLabel = kNNClassification_openmp(trainingSet, unknownPoint, k);
	std::cout << "Bilinmeyen nokta etiketi:(OpenMp) " << predictedLabel << std::endl;
	TIMERSTOP(kNNClassification_openmp)

		return 0;
}
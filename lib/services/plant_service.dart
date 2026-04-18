import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:flutter/material.dart';
import '../models/plant.dart';

class PlantService with ChangeNotifier {
  List<Plant> _availablePlants = [];
  List<Plant> _userPlants = [];

  List<Plant> get availablePlants => _availablePlants;
  List<Plant> get userPlants => _userPlants;

  PlantService() {
    loadPlants();
  }

  Future<void> loadPlants() async {
    final jsonString =
        await rootBundle.loadString('assets/plants_data.json');
    final List<dynamic> jsonList = json.decode(jsonString);
    _availablePlants = jsonList.map((json) => Plant.fromJson(json)).toList();
    notifyListeners();
  }

  void addUserPlant(Plant plant) {
    _userPlants.add(plant);
    notifyListeners();
  }

  void removeUserPlant(int id) {
    _userPlants.removeWhere((plant) => plant.id == id);
    notifyListeners();
  }
}
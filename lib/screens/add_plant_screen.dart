import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/plant.dart';
import '../services/plant_service.dart';

class AddPlantScreen extends StatefulWidget {
  const AddPlantScreen({super.key});

  @override
  State<AddPlantScreen> createState() => _AddPlantScreenState();
}

class _AddPlantScreenState extends State<AddPlantScreen> {
  String _searchQuery = '';
  int _selectedCategoryIndex = 0;
  final List<String> _categories = ['All', 'Vegetable', 'Fruit', 'Herb', 'Tree'];

  @override
  Widget build(BuildContext context) {
    final plantService = Provider.of<PlantService>(context);
    final availablePlants = plantService.availablePlants;

    // Filter by category
    List<Plant> categoryPlants;
    if (_selectedCategoryIndex == 0) {
      categoryPlants = availablePlants;
    } else {
      final selectedCategory = _categories[_selectedCategoryIndex].toLowerCase();
      categoryPlants = availablePlants
          .where((plant) => plant.category.toLowerCase() == selectedCategory)
          .toList();
    }

    // Then filter by search query
    final filteredPlants = categoryPlants.where((plant) {
      final name = plant.name.toLowerCase();
      final sciName = plant.scientificName.toLowerCase();
      final query = _searchQuery.toLowerCase();
      return name.contains(query) || sciName.contains(query);
    }).toList();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Add a Plant'),
      ),
      body: Column(
        children: [
          // Search bar
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            child: TextField(
              decoration: InputDecoration(
                hintText: 'Search plants...',
                prefixIcon: const Icon(Icons.search),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                filled: true,
                fillColor: Colors.grey.shade50,
              ),
              onChanged: (value) {
                setState(() {
                  _searchQuery = value;
                });
              },
            ),
          ),
          // Category chips
          SizedBox(
            height: 60,
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(horizontal: 12),
              itemCount: _categories.length,
              itemBuilder: (context, index) {
                return Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 4),
                  child: ChoiceChip(
                    label: Text(_categories[index]),
                    selected: _selectedCategoryIndex == index,
                    onSelected: (selected) {
                      setState(() {
                        _selectedCategoryIndex = index;
                      });
                    },
                    selectedColor: Colors.green.shade200,
                  ),
                );
              },
            ),
          ),
          // Plant list
          Expanded(
            child: availablePlants.isEmpty
                ? const Center(child: CircularProgressIndicator())
                : filteredPlants.isEmpty
                    ? const Center(
                        child: Text(
                          'No plants found',
                          style: TextStyle(fontSize: 18, color: Colors.grey),
                        ),
                      )
                    : ListView.builder(
                        padding: const EdgeInsets.all(16),
                        itemCount: filteredPlants.length,
                        itemBuilder: (context, index) {
                          final plant = filteredPlants[index];
                          return _PlantItem(
                            plant: plant,
                            onTap: () {
                              // Add this plant to user's list
                              final newPlant = Plant(
                                id: plant.id,
                                name: plant.name,
                                scientificName: plant.scientificName,
                                imageUrl: plant.imageUrl,
                                description: plant.description,
                                category: plant.category,
                                stages: plant.stages,
                                addedDate: DateTime.now(),
                              );
                              plantService.addUserPlant(newPlant);
                              Navigator.pop(context);
                            },
                          );
                        },
                      ),
          ),
        ],
      ),
    );
  }
}

class _PlantItem extends StatelessWidget {
  final Plant plant;
  final VoidCallback onTap;

  const _PlantItem({
    required this.plant,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: Colors.green.shade100,
          child: const Icon(Icons.eco, color: Colors.green),
        ),
        title: Text(
          plant.name,
          style: const TextStyle(fontWeight: FontWeight.bold),
        ),
        subtitle: Text(plant.scientificName),
        trailing: const Icon(Icons.add_circle_outline),
        onTap: onTap,
      ),
    );
  }
}
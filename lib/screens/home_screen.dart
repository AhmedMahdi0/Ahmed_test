import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/plant.dart';
import '../services/plant_service.dart';
import 'add_plant_screen.dart';
import 'plant_detail_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final plantService = Provider.of<PlantService>(context);
    final userPlants = plantService.userPlants;

    return Scaffold(
      appBar: AppBar(
        title: const Text('🌱 Plant Growth Tracker'),
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: () {
              showAboutDialog(
                context: context,
                applicationName: 'Plant Growth Tracker',
                applicationVersion: '1.0.0',
                applicationLegalese: 'Track your plants\' growth stages',
              );
            },
          ),
        ],
      ),
      body: userPlants.isEmpty
          ? const _EmptyState()
          : ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: userPlants.length,
              itemBuilder: (context, index) {
                final plant = userPlants[index];
                final days = plant.getDaysSinceAdded();
                final currentStage = plant.getCurrentStage();
                final progress = plant.getProgressPercentage();
                return _PlantCard(
                  plant: plant,
                  days: days,
                  currentStage: currentStage,
                  progress: progress,
                );
              },
            ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          Navigator.push(
            context,
            MaterialPageRoute(builder: (context) => const AddPlantScreen()),
          );
        },
        child: const Icon(Icons.add),
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Image.asset(
            'assets/empty_illustration.png',
            height: 200,
            width: 200,
            errorBuilder: (_, __, ___) => const Icon(
              Icons.eco,
              size: 120,
              color: Colors.grey,
            ),
          ),
          const SizedBox(height: 24),
          const Text(
            'No plants yet',
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 12),
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 48),
            child: Text(
              'Add your first plant to track its growth stages over time.',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.grey, fontSize: 16),
            ),
          ),
          const SizedBox(height: 24),
          ElevatedButton.icon(
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => const AddPlantScreen()),
              );
            },
            icon: const Icon(Icons.add),
            label: const Text('Add a Plant'),
          ),
        ],
      ),
    );
  }
}

class _PlantCard extends StatelessWidget {
  final Plant plant;
  final int days;
  final GrowthStage? currentStage;
  final double progress;

  const _PlantCard({
    required this.plant,
    required this.days,
    required this.currentStage,
    required this.progress,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: InkWell(
        onTap: () {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => PlantDetailScreen(plant: plant),
            ),
          );
        },
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  CircleAvatar(
                    backgroundColor: Colors.green.shade100,
                    radius: 30,
                    child: const Icon(Icons.eco, size: 30, color: Colors.green),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          plant.name,
                          style: const TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        Text(
                          plant.scientificName,
                          style: const TextStyle(
                            fontSize: 14,
                            color: Colors.grey,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Day $days',
                          style: TextStyle(
                            fontSize: 16,
                            color: Colors.green.shade700,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.more_vert),
                    onPressed: () {},
                  ),
                ],
              ),
              const SizedBox(height: 16),
              if (currentStage != null)
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      currentStage!.title,
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      currentStage!.description,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(color: Colors.grey, fontSize: 14),
                    ),
                  ],
                ),
              const SizedBox(height: 16),
              LinearProgressIndicator(
                value: progress,
                backgroundColor: Colors.grey.shade300,
                color: Colors.green,
                minHeight: 8,
                borderRadius: BorderRadius.circular(4),
              ),
              const SizedBox(height: 8),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    '${(progress * 100).toStringAsFixed(1)}% complete',
                    style: const TextStyle(fontSize: 12, color: Colors.grey),
                  ),
                  Text(
                    'Stage ${currentStage?.stageId ?? 0} of ${plant.stages.length}',
                    style: const TextStyle(fontSize: 12, color: Colors.grey),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
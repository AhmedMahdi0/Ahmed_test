import 'dart:convert';

class Plant {
  final int id;
  final String name;
  final String scientificName;
  final String imageUrl;
  final String description;
  final String category;
  final List<GrowthStage> stages;
  final DateTime addedDate;

  Plant({
    required this.id,
    required this.name,
    required this.scientificName,
    required this.imageUrl,
    required this.description,
    required this.stages,
    required this.addedDate,
    this.category = 'other',
  });

  factory Plant.fromJson(Map<String, dynamic> json) {
    return Plant(
      id: json['id'],
      name: json['name'],
      scientificName: json['scientific_name'],
      imageUrl: json['image_url'],
      description: json['description'],
      category: json['category'] ?? 'other',
      stages: (json['stages'] as List)
          .map((stage) => GrowthStage.fromJson(stage))
          .toList(),
      addedDate: DateTime.now(),
    );
  }

  int getDaysSinceAdded() {
    final now = DateTime.now();
    return now.difference(addedDate).inDays;
  }

  GrowthStage? getCurrentStage() {
    final days = getDaysSinceAdded();
    for (var stage in stages) {
      final range = stage.dayRange.split('-');
      if (range.length == 2) {
        final start = int.tryParse(range[0]) ?? 0;
        final end = int.tryParse(range[1]) ?? 0;
        if (days >= start && days <= end) {
          return stage;
        }
      }
    }
    return null;
  }

  double getProgressPercentage() {
    final days = getDaysSinceAdded();
    // Find total possible days (max end range)
    int maxDays = 0;
    for (var stage in stages) {
      final range = stage.dayRange.split('-');
      if (range.length == 2) {
        final end = int.tryParse(range[1]) ?? 0;
        if (end > maxDays) maxDays = end;
      }
    }
    if (maxDays == 0) return 0.0;
    return (days / maxDays).clamp(0.0, 1.0);
  }
}

class GrowthStage {
  final int stageId;
  final String dayRange;
  final String title;
  final String description;

  GrowthStage({
    required this.stageId,
    required this.dayRange,
    required this.title,
    required this.description,
  });

  factory GrowthStage.fromJson(Map<String, dynamic> json) {
    return GrowthStage(
      stageId: json['stage_id'],
      dayRange: json['day_range'],
      title: json['title'],
      description: json['description'],
    );
  }
}
import React from "react";
import {
  StyleSheet,
  Text,
  View,
  FlatList,
  Image,
  TouchableOpacity,
  Dimensions,
} from "react-native";

const recommendations = [
  {
    id: "1",
    title: "Learn React Native",
    description: "Start your journey with React Native.",
    image: require("../../assets/images/image1.png"),
  },
  {
    id: "2",
    title: "Master JavaScript",
    description: "Enhance your JavaScript skills.",
    image: require("../../assets/images/image2.png"),
  },
  {
    id: "3",
    title: "Advanced React Patterns",
    description: "Learn advanced patterns in React.",
    image: require("../../assets/images/image3.png"),
  },

  // Add more items as needed
];

const RecommendationCard = ({ item }) => (
  <TouchableOpacity style={styles.card}>
    <Image source={item.image} style={styles.cardImage} />
    <View style={styles.cardContent}>
      <Text style={styles.cardTitle}>{item.title}</Text>
      <Text style={styles.cardDescription}>{item.description}</Text>
    </View>
  </TouchableOpacity>
);

const RecommendationsScreen = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.header}>Recommended for You</Text>
      <FlatList
        data={recommendations}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => <RecommendationCard item={item} />}
        contentContainerStyle={styles.list}
        showsVerticalScrollIndicator={false}
      />
    </View>
  );
};

export default RecommendationsScreen;
const { width } = Dimensions.get("window");

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#2f2e41", // Secondary color
    paddingHorizontal: 20,
    paddingTop: 40,
  },
  header: {
    fontSize: 24,
    color: "#fff",
    fontWeight: "bold",
    marginBottom: 20,
  },
  list: {
    paddingBottom: 20,
  },
  card: {
    backgroundColor: "#fff",
    borderRadius: 12,
    marginBottom: 16,
    overflow: "hidden",
    flexDirection: "row",
    elevation: 3,
  },
  cardImage: {
    width: 100,
    height: 100,
  },
  cardContent: {
    flex: 1,
    padding: 12,
    justifyContent: "center",
  },
  cardTitle: {
    fontSize: 18,
    color: "#8117c1", // Primary color
    fontWeight: "bold",
    marginBottom: 4,
  },
  cardDescription: {
    fontSize: 14,
    color: "#333",
  },
});

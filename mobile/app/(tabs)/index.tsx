import React from "react";
import {
  StyleSheet,
  Text,
  View,
  TextInput,
  FlatList,
  TouchableOpacity,
  Dimensions,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";

const features = [
  { id: "1", name: "Explore", icon: "compass-outline" },
  { id: "2", name: "Messages", icon: "chatbubble-outline" },
  { id: "3", name: "Notifications", icon: "notifications-outline" },
  { id: "4", name: "Settings", icon: "settings-outline" },
  // Add more features as needed
];

const FeatureCard = ({ item }) => (
  <TouchableOpacity style={styles.card}>
    <Ionicons name={item.icon} size={32} color="#8117c1" />
    <Text style={styles.cardText}>{item.name}</Text>
  </TouchableOpacity>
);

const HomeScreen = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.header}>Welcome Back!</Text>
      <TextInput
        style={styles.searchInput}
        placeholder="Search..."
        placeholderTextColor="#ccc"
      />
      <FlatList
        data={features}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => <FeatureCard item={item} />}
        numColumns={2}
        columnWrapperStyle={styles.row}
        contentContainerStyle={styles.list}
        showsVerticalScrollIndicator={false}
      />
    </View>
  );
};

export default HomeScreen;

const { width } = Dimensions.get("window");
const cardSize = (width - 60) / 2;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#2f2e41", // Secondary color
    paddingHorizontal: 20,
    paddingTop: 40,
  },
  header: {
    fontSize: 28,
    color: "#fff",
    fontWeight: "bold",
    marginBottom: 20,
  },
  searchInput: {
    backgroundColor: "#fff",
    borderRadius: 8,
    paddingHorizontal: 15,
    paddingVertical: 10,
    fontSize: 16,
    marginBottom: 20,
  },
  list: {
    paddingBottom: 20,
  },
  row: {
    justifyContent: "space-between",
    marginBottom: 20,
  },
  card: {
    backgroundColor: "#fff",
    borderRadius: 12,
    width: cardSize,
    height: cardSize,
    alignItems: "center",
    justifyContent: "center",
    elevation: 3,
  },
  cardText: {
    marginTop: 10,
    fontSize: 16,
    color: "#8117c1", // Primary color
    fontWeight: "bold",
  },
});


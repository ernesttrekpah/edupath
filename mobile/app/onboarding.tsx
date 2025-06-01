import React from "react";
import { View, Text, Image, StyleSheet, Dimensions } from "react-native";
import Onboarding from "react-native-onboarding-swiper";
import { useRouter } from "expo-router";

const { width: windowWidth, height: windowHeight } = Dimensions.get("window");

const styles = StyleSheet.create({
  image: {
    width: windowWidth * 0.8, // 80% of the screen width
    height: windowHeight * 0.4, // 40% of the screen height
    resizeMode: "contain",
  },
});

export default function OnboardingScreen() {
  const router = useRouter();

  return (
    <Onboarding
      onDone={() => router.replace("auth/signup")}
      onSkip={() => router.replace("auth/signup")}
      showDone={true}
      pages={[
        {
          backgroundColor: "#fff",
          image: (
            <Image
              source={require("../assets/images/image1.png")}
              style={styles.image}
            />
          ),
          title: "Welcome",
          subtitle: "Discover new features.",
        },
        {
          backgroundColor: "#fff",
          image: (
            <Image
              source={require("../assets/images/image2.png")}
              style={styles.image}
            />
          ),
          title: "Stay Connected",
          subtitle: "Connect with people around the world.",
        },
        {
          backgroundColor: "#fff",
          image: (
            <Image
              source={require("../assets/images/image3.png")}
              style={styles.image}
            />
          ),
          title: "Get Started",
          subtitle: "Let’s take you to the app!",
        },
      ]}
    />
  );
}

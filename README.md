# Embodied Coffee Coach

A small experimental embodied AI prototype that guides users through an AeroPress coffee brewing workflow using physical interaction, webcam sensing, hand tracking, and a local vision language model.

## Demo

Video demo: https://youtu.be/VdKNlfER7sE

## Overview

Embodied Coffee Coach explores hands free interaction for physical coffee brewing tasks.

Instead of using a keyboard, mouse, or touchscreen, the user interacts with the system through natural actions such as pouring coffee beans, grinding, adding water, stirring, waiting, and pressing an AeroPress.

The system observes these actions through the MacBook webcam and provides visual and spoken guidance throughout the brewing process.

## How It Works

The system combines two sensing and reasoning components.

MediaPipe Hand Landmarker tracks hand movement and extracts motion information such as movement intensity, repetitive motion, and movement direction.

A local Qwen3 VL model running through Ollama analyzes temporally separated webcam frames together with the hand motion information. It reasons about which objects are visible and what physical action the user is performing.

An interaction state manager then uses these AI interpretations, confidence values, object constraints, timing information, and repeated observations to determine whether the workflow should progress.

## System Architecture

```text
Physical Interaction
        ↓
MacBook Webcam
        ↓
MediaPipe Hand Tracking
        ↓
Temporal Webcam Observations
        ↓
Qwen3 VL
        ↓
Object and Action Reasoning
        ↓
Interaction State Manager
        ↓
Visual and Spoken Guidance
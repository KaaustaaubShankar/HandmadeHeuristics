from manim import *

class KNNVisualization(Scene):
    def construct(self):
        # Create the axes
        axes = Axes(
            x_range=[0, 10, 1], y_range=[0, 10, 1],
            axis_config={"color": BLUE},
            x_length=7, y_length=7
        )
        labels = axes.get_axis_labels(x_label="X", y_label="Y")
        self.play(Create(axes), Write(labels))

        # Sample data points (features and labels) including three groups: A, B, and C
        data_points = [
            (2, 3, "A"), (4, 8, "A"), (6, 1, "B"), (7, 6, "B"),
            (5, 4, "A"), (8, 3, "B"), (3, 7, "A"), (9, 8, "B"),
            (1, 5, "C"), (6, 9, "C"), (7, 2, "C"), (3, 2, "C")
        ]

        # Colors for classes
        class_colors = {
            "A": RED,
            "B": GREEN,
            "C": BLUE
        }

        # Plot data points with class labels
        point_mobjects = []
        for x, y, label in data_points:
            color = class_colors[label]
            point = Dot(axes.coords_to_point(x, y), color=color)
            point_label = Text(label).scale(0.5).next_to(point, UP)
            self.play(Create(point), Write(point_label))
            point_mobjects.append(point)

        # Annotate the scene explaining the classes
        class_a_text = Text("Class A (Red)", color=RED).scale(0.6).to_corner(UL, buff=1)
        class_b_text = Text("Class B (Green)", color=GREEN).scale(0.6).next_to(class_a_text, DOWN, aligned_edge=LEFT)
        class_c_text = Text("Class C (Blue)", color=BLUE).scale(0.6).next_to(class_b_text, DOWN, aligned_edge=LEFT)
        self.play(Write(class_a_text), Write(class_b_text), Write(class_c_text))

        # Test point for which we want to predict the class
        test_point_coords = (6, 5)
        test_point = Dot(axes.coords_to_point(*test_point_coords), color=YELLOW, radius=0.15)
        test_label = Text("Test Point").scale(0.5).next_to(test_point, UP)
        self.play(Create(test_point), Write(test_label))

        # Explanation of the k value
        k_value_text = Text("Finding the 3 nearest neighbors (k=3)", font_size=24, color=BLUE).to_edge(UP)
        self.play(Write(k_value_text))

        # Calculate the distances from the test point to all other points
        k = 3
        distances = []
        for i, (x, y, _) in enumerate(data_points):
            distance = ((x - test_point_coords[0])**2 + (y - test_point_coords[1])**2)**0.5
            distances.append((distance, i))
        
        # Sort the distances and identify the k nearest neighbors
        distances.sort(key=lambda d: d[0])
        nearest_neighbors = [data_points[i] for _, i in distances[:k]]
        
        # Highlight and annotate the nearest neighbors
        neighbor_annotations = []
        for x, y, label in nearest_neighbors:
            neighbor_point = Dot(axes.coords_to_point(x, y), color=class_colors[label], radius=0.2)
            self.play(TransformFromCopy(Dot(axes.coords_to_point(x, y), color=class_colors[label]), neighbor_point))
            annotation = Text(f"Distance: {((x - test_point_coords[0])**2 + (y - test_point_coords[1])**2)**0.5:.2f}").scale(0.4).next_to(neighbor_point, DOWN)
            self.play(Write(annotation))
            neighbor_annotations.append(annotation)
            self.wait(0.5)

        # Draw lines connecting the test point to the nearest neighbors
        for x, y, _ in nearest_neighbors:
            line = Line(test_point.get_center(), axes.coords_to_point(x, y), color=YELLOW)
            self.play(Create(line))
            self.wait(0.5)

        # Calculate the predicted class based on the nearest neighbors
        predicted_label = max(set([label for _, _, label in nearest_neighbors]), key=[label for _, _, label in nearest_neighbors].count)

        # Display the predicted class
        prediction_text = Text(f"Predicted Class: {predicted_label}", color=YELLOW, font_size=28).to_edge(DOWN)
        prediction_explanation = Text(f"The majority of the {k} nearest neighbors are class {predicted_label}.", font_size=24).next_to(prediction_text, UP)
        self.play(Write(prediction_text), Write(prediction_explanation))

        # Final animation to emphasize the test point and prediction
        self.play(test_point.animate.set_color(class_colors[predicted_label]))
        self.play(FadeOut(k_value_text), *[FadeOut(ann) for ann in neighbor_annotations], FadeOut(labels))

        # Final wait before ending the scene
        self.wait(3)

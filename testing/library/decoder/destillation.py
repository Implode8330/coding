import keras
import tensorflow as tf

class Teacher_Student_Training(keras.Model):
    def __init__(self, student):
        super(Teacher_Student_Training, self).__init__()
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.8,
    ):

        super(Teacher_Student_Training, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn        = student_loss_fn
        self.distillation_loss_fn   = distillation_loss_fn
        self.alpha                  = alpha

    def train_step(self, data):
        # Unpack data
        x, y, y_teacher = data[0]
        # Forward pass of teacher
        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student.model(x)
            # Compute losses
            student_loss        = self.student_loss_fn(y, student_predictions)
            distillation_loss   = self.distillation_loss_fn(y_teacher, student_predictions)
            loss = self.alpha * student_loss + ((1 - self.alpha) * distillation_loss)
        # Compute gradients
        trainable_vars  = self.student.trainable_variables
        gradients       = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y, y_teacher = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

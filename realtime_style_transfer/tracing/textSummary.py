import tensorflow as tf


def capture_model_summary(model: tf.keras.Model, detailed=False) -> str:
    summary_lines = ""

    def append_line(line):
        nonlocal summary_lines
        summary_lines += f"{line}\n"

    model.summary(line_length=240, show_trainable=True, expand_nested=detailed, print_fn=append_line)

    return summary_lines

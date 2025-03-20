from langchain_core.tools import tool
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

@tool
def generate_visualization(data: list, chart_type: str = "bar") -> str:
    """
    Generate a visualization from structured data.
    
    Args:
        data (list): A list of dictionaries representing the query result.
        chart_type (str): Type of chart to generate (bar, line, scatter, pie).
        
    Returns:
        str: A base64-encoded image of the visualization.
    """
    if not data or not isinstance(data, list) or not isinstance(data[0], dict):
        return "Invalid data format for visualization."

    # Convert result to DataFrame
    df = pd.DataFrame(data)
    
    # Ensure at least two columns exist
    if df.shape[1] < 2:
        return "Not enough data to generate a chart."
    
    x_values = df.iloc[:, 0].astype(str).tolist()  # Convert first column to string (labels)
    y_values = df.iloc[:, 1].tolist()  # Second column as numerical values

    # Create plot based on the requested chart type
    plt.figure(figsize=(8, 5))
    if chart_type == "bar":
        plt.bar(x_values, y_values, color='blue')
    elif chart_type == "line":
        plt.plot(x_values, y_values, marker='o', linestyle='-', color='green')
    elif chart_type == "scatter":
        plt.scatter(x_values, y_values, color='red')
    elif chart_type == "pie":
        plt.pie(y_values, labels=x_values, autopct='%1.1f%%', colors=['blue', 'orange', 'green', 'red'])
    else:
        return "Unsupported chart type. Use bar, line, scatter, or pie."

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(f"{chart_type.capitalize()} Chart")
    plt.xticks(rotation=45)

    # Save plot to a base64-encoded string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", bbox_inches="tight")
    plt.close()
    
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

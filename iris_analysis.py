"""
Iris Dataset Analysis and Visualization

This script performs a comprehensive analysis of the Iris dataset, including:
1. Data loading and preprocessing
2. Basic statistical analysis
3. Data visualization using various plots
4. Error handling for robust execution

The script creates multiple visualizations:
- Line chart of feature measurements
- Bar chart of mean measurements by species
- Histogram of feature distributions
- Scatter plot of feature relationships
- Box plot of feature distributions
- Correlation heatmap

Requirements:
    - Python 3.x
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scikit-learn

Usage:
    python iris_analysis.py

Output:
    - Console output with statistical analysis
    - 'iris_analysis.png' file containing all visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import sys
import os

def load_data():
    """
    Load the Iris dataset and convert it to a pandas DataFrame.
    
    This function:
    1. Loads the Iris dataset from scikit-learn
    2. Creates a pandas DataFrame with feature names
    3. Adds species labels to the DataFrame
    4. Performs initial data exploration
    
    Returns:
        pandas.DataFrame: DataFrame containing Iris dataset with species labels
    
    Raises:
        ImportError: If required libraries are not installed
        Exception: For any other errors during data loading
    """
    try:
        # Load the dataset
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = iris.target_names[iris.target]
        
        # Display initial data exploration
        print("\nFirst 5 rows of the dataset:")
        print("-" * 50)
        print(df.head())
        
        print("\nDataset Information:")
        print("-" * 50)
        print(df.info())
        
        print("\nMissing Values:")
        print("-" * 50)
        print(df.isnull().sum())
        
        return df
    except ImportError:
        print("Error: Required libraries not found. Please install required packages using 'pip install -r requirements.txt'")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

def analyze_data(df):
    """
    Perform comprehensive statistical analysis on the Iris dataset.
    
    This function:
    1. Displays dataset information (samples, features)
    2. Shows basic statistics (mean, std, min, max, etc.)
    3. Displays species distribution
    4. Performs group analysis by species
    
    Args:
        df (pandas.DataFrame): Input DataFrame containing Iris data
    
    Raises:
        AttributeError: If DataFrame structure is invalid
        Exception: For any other errors during analysis
    """
    try:
        # Display basic dataset information
        print("\nDataset Information:")
        print("-" * 50)
        print(f"Number of samples: {len(df)}")
        print(f"Number of features: {len(df.columns) - 1}")  # Excluding species column
        print("\nFeature Names:")
        print(df.columns[:-1].tolist())
        
        # Display statistical summary
        print("\nBasic Statistics:")
        print("-" * 50)
        print(df.describe())
        
        # Display species distribution
        print("\nSpecies Distribution:")
        print("-" * 50)
        print(df['species'].value_counts())
        
        # Group analysis by species
        print("\nMean Measurements by Species:")
        print("-" * 50)
        species_means = df.groupby('species').mean()
        print(species_means)
        
        # Correlation analysis
        print("\nFeature Correlations:")
        print("-" * 50)
        correlation = df.iloc[:, :-1].corr()
        print(correlation)
        
    except AttributeError:
        print("Error: Invalid DataFrame structure. Expected columns not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during data analysis: {str(e)}")
        sys.exit(1)

def create_visualizations(df):
    """
    Create and save various visualizations of the Iris dataset.
    
    This function creates four different plots:
    1. Line chart showing feature measurements across samples
    2. Bar chart showing mean measurements by species
    3. Histogram showing feature distributions
    4. Scatter plot showing relationship between features
    
    Args:
        df (pandas.DataFrame): Input DataFrame containing Iris data
    
    Raises:
        KeyError: If required columns are missing
        ValueError: If data format is invalid
        PermissionError: If unable to save visualization file
        Exception: For any other errors during visualization
    """
    try:
        # Create a figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Iris Dataset Analysis', fontsize=16, y=1.02)
        
        # 1. Line Chart - Feature measurements across samples
        axes[0, 0].plot(df['sepal length (cm)'].values, label='Sepal Length', color='blue')
        axes[0, 0].plot(df['petal length (cm)'].values, label='Petal Length', color='red')
        axes[0, 0].set_title('Feature Measurements Across Samples')
        axes[0, 0].set_xlabel('Sample Index')
        axes[0, 0].set_ylabel('Measurement (cm)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Bar Chart - Mean measurements by species
        species_means = df.groupby('species').mean()
        species_means.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Mean Measurements by Species')
        axes[0, 1].set_xlabel('Species')
        axes[0, 1].set_ylabel('Measurement (cm)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True)
        
        # 3. Histogram - Feature distributions
        sns.histplot(data=df, x='sepal length (cm)', hue='species', 
                    multiple="stack", ax=axes[1, 0])
        axes[1, 0].set_title('Sepal Length Distribution by Species')
        axes[1, 0].set_xlabel('Sepal Length (cm)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True)
        
        # 4. Scatter Plot - Relationship between features
        sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', 
                       hue='species', palette='deep', ax=axes[1, 1])
        axes[1, 1].set_title('Sepal Length vs Petal Length')
        axes[1, 1].set_xlabel('Sepal Length (cm)')
        axes[1, 1].set_ylabel('Petal Length (cm)')
        axes[1, 1].grid(True)
        
        # Adjust layout and save the figure
        plt.tight_layout()
        
        # Check if we have write permissions before saving
        try:
            plt.savefig('iris_analysis.png', bbox_inches='tight', dpi=300)
        except PermissionError:
            print("Error: No permission to save the visualization file.")
            sys.exit(1)
        except Exception as e:
            print(f"Error saving visualization: {str(e)}")
            sys.exit(1)
        finally:
            plt.close()
            
    except KeyError as e:
        print(f"Error: Required column not found in data: {str(e)}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid data format: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        sys.exit(1)

def main():
    """
    Main function to orchestrate the Iris dataset analysis.
    
    This function:
    1. Loads the Iris dataset
    2. Performs statistical analysis
    3. Creates and saves visualizations
    
    Raises:
        KeyboardInterrupt: If program is interrupted by user
        Exception: For any unexpected errors
    """
    try:
        # Load the data
        print("Loading Iris dataset...")
        df = load_data()
        
        # Perform analysis
        print("\nPerforming data analysis...")
        analyze_data(df)
        
        # Create visualizations
        print("\nCreating visualizations...")
        create_visualizations(df)
        print("\nVisualizations have been saved as 'iris_analysis.png'")
        
        print("\nAnalysis complete!")
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
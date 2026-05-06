"""
Breast Cancer Wisconsin Statistical Analysis Project
Author: Data Analyst
Date: 2024
Description: Comprehensive statistical analysis of breast cancer dataset
"""

import pandas as pd
import numpy as np
from functools import reduce
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class BreastCancerAnalyzer:
    """
    Main class for Breast Cancer Wisconsin dataset analysis
    """
    
    def __init__(self, file_path):
        """
        Initialize the analyzer with dataset file path
        
        Args:
            file_path (str): Path to the CSV file
        """
        self.file_path = file_path
        self.df = None
        
    def load_data(self):
        """
        Step 1: Load CSV file with exception handling
        """
        try:
            print("="*60)
            print("STEP 1: Loading Dataset")
            print("="*60)
            
            # Try to read the CSV file
            self.df = pd.read_csv(self.file_path)
            
            # Drop the unnamed index column if it exists
            if 'Unnamed: 32' in self.df.columns:
                self.df = self.df.drop('Unnamed: 32', axis=1)
            
            print(f"✅ Successfully loaded dataset with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
            return True
            
        except FileNotFoundError:
            print(f"❌ Error: File '{self.file_path}' not found. Please check the file path.")
            return False
        except PermissionError:
            print(f"❌ Error: Permission denied to read '{self.file_path}'.")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def explore_data(self):
        """
        Step 2: Data exploration - display first/last rows and info
        """
        print("\n" + "="*60)
        print("STEP 2: Data Exploration")
        print("="*60)
        
        # Display first 10 rows
        print("\n📊 First 10 rows of the dataset:")
        print(self.df.head(10))
        
        # Display last 5 rows
        print("\n📊 Last 5 rows of the dataset:")
        print(self.df.tail(5))
        
        # Display DataFrame info
        print("\n📊 DataFrame Information:")
        print(self.df.info())
        
        # Use for-loop with relational operators to print samples
        print("\n📊 Samples with radius_mean > 15 OR area_mean > 700:")
        count = 0
        for idx, row in self.df.iterrows():
            if row['radius_mean'] > 15 or row['area_mean'] > 700:
                print(f"  Sample {idx}: Diagnosis={row['diagnosis']}, "
                      f"radius_mean={row['radius_mean']:.2f}, area_mean={row['area_mean']:.2f}")
                count += 1
                if count >= 10:  # Limit output to first 10 samples
                    print("  ... (showing first 10 samples only)")
                    break
    
    def handle_missing_values(self):
        """
        Step 3: Handle missing values using manual calculations
        """
        print("\n" + "="*60)
        print("STEP 3: Handling Missing Values")
        print("="*60)
        
        # Check for missing values
        missing_before = self.df.isnull().sum()
        missing_cols = missing_before[missing_before > 0]
        
        if len(missing_cols) == 0:
            print("✅ No missing values found in the dataset")
            return
        
        print(f"Found missing values in columns: {list(missing_cols.index)}")
        
        # Function to calculate column mean manually
        def manual_mean(column_data):
            total = 0
            count = 0
            for value in column_data:
                if pd.notna(value):
                    total += value
                    count += 1
            return total / count if count > 0 else 0
        
        # Fill missing values with column means
        for col in missing_cols.index:
            col_mean = manual_mean(self.df[col])
            print(f"Filling missing values in '{col}' with mean: {col_mean:.4f}")
            
            # Manual filling using loop
            for idx in self.df.index:
                if pd.isna(self.df.loc[idx, col]):
                    self.df.loc[idx, col] = col_mean
        
        # Verify no missing values remain
        missing_after = self.df.isnull().sum().sum()
        print(f"✅ Missing values after handling: {missing_after}")
    
    def create_new_columns(self):
        """
        Step 4: Create new calculated columns
        """
        print("\n" + "="*60)
        print("STEP 4: Creating New Columns")
        print("="*60)
        
        # i) Worst_Area_Mean
        self.df['Worst_Area_Mean'] = (self.df['area_worst'] + self.df['area_mean']) / 2
        print("✅ Created column: Worst_Area_Mean")
        
        # ii) Radius_Texture_Index
        self.df['Radius_Texture_Index'] = self.df['radius_mean'] * self.df['texture_mean']
        print("✅ Created column: Radius_Texture_Index")
        
        # iii) MalignancyRiskScore using if-elif-else
        def calculate_risk_score(row):
            score = 0
            # Multiple feature thresholds
            if row['radius_mean'] > 17:
                score += 3
            elif row['radius_mean'] > 14:
                score += 2
            else:
                score += 1
            
            if row['area_mean'] > 800:
                score += 3
            elif row['area_mean'] > 500:
                score += 2
            else:
                score += 1
            
            if row['concavity_mean'] > 0.1:
                score += 3
            elif row['concavity_mean'] > 0.05:
                score += 2
            else:
                score += 1
            
            return score
        
        self.df['MalignancyRiskScore'] = self.df.apply(calculate_risk_score, axis=1)
        print("✅ Created column: MalignancyRiskScore")
        
        # Display sample of new columns
        print("\n📊 Sample of new columns (first 5 rows):")
        new_cols = ['Worst_Area_Mean', 'Radius_Texture_Index', 'MalignancyRiskScore']
        print(self.df[new_cols].head())
    
    def radius_mean_checks(self):
        """
        Step 5: Radius > Mean checks using relational and bitwise operators
        """
        print("\n" + "="*60)
        print("STEP 5: Radius and Area Mean Checks")
        print("="*60)
        
        # Calculate overall means
        overall_radius_mean = self.df['radius_mean'].mean()
        overall_area_worst_mean = self.df['area_worst'].mean()
        
        print(f"Overall mean of radius_mean: {overall_radius_mean:.4f}")
        print(f"Overall mean of area_worst: {overall_area_worst_mean:.4f}")
        
        # Method 1: Using pandas with bitwise operators
        flag_pandas = (self.df['radius_mean'] > overall_radius_mean) & \
                      (self.df['area_worst'] > overall_area_worst_mean)
        
        print(f"\n📊 Pandas method - Samples meeting criteria: {flag_pandas.sum()}")
        
        # Method 2: Manual for-loop with if conditions
        manual_flags = []
        for idx in self.df.index:
            if self.df.loc[idx, 'radius_mean'] > overall_radius_mean and \
               self.df.loc[idx, 'area_worst'] > overall_area_worst_mean:
                manual_flags.append(idx)
        
        print(f"Manual method - Samples meeting criteria: {len(manual_flags)}")
        
        # Display some flagged samples
        print("\n📊 First 5 flagged samples:")
        flagged_samples = self.df[flag_pandas].head()
        for idx in flagged_samples.index:
            print(f"  Sample {idx}: radius_mean={self.df.loc[idx, 'radius_mean']:.2f}, "
                  f"area_worst={self.df.loc[idx, 'area_worst']:.2f}, "
                  f"diagnosis={self.df.loc[idx, 'diagnosis']}")
    
    def custom_reduce_calculations(self):
        """
        Step 6: Using reduce() for custom mean/median calculations
        """
        print("\n" + "="*60)
        print("STEP 6: Custom Mean/Median Calculations using reduce()")
        print("="*60)
        
        features = ['radius_mean', 'area_mean', 'compactness_mean', 'concavity_mean']
        
        # Using reduce to calculate mean
        def sum_reducer(acc, val):
            return acc + val
        
        def count_reducer(acc, val):
            return acc + 1
        
        for feature in features:
            # Calculate mean using reduce
            total = reduce(sum_reducer, self.df[feature], 0)
            count = reduce(count_reducer, self.df[feature], 0)
            reduce_mean = total / count
            
            # Manual mean calculation with loop for comparison
            manual_total = 0
            manual_count = 0
            for value in self.df[feature]:
                manual_total += value
                manual_count += 1
            manual_mean = manual_total / manual_count
            
            # Median calculation using reduce and sort
            sorted_values = sorted(self.df[feature])
            n = len(sorted_values)
            if n % 2 == 0:
                median = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
            else:
                median = sorted_values[n//2]
            
            print(f"\n📊 {feature}:")
            print(f"  Reduce Mean: {reduce_mean:.4f}")
            print(f"  Manual Mean: {manual_mean:.4f}")
            print(f"  Median: {median:.4f}")
            print(f"  Difference in means: {abs(reduce_mean - manual_mean):.10f}")
    
    def recursive_fractal_dimension(self, index=0, accumulator=0):
        """
        Step 7: Recursive function to approximate fractal dimension
        
        Args:
            index (int): Current row index
            accumulator (float): Accumulator for fractal approximations
            
        Returns:
            tuple: (average_approximation, total_samples)
        """
        # Base case: reached end of dataset
        if index >= len(self.df):
            if index == 0:
                return 0, 0
            return accumulator / index, index
        
        # Get perimeter and area features
        perimeter = self.df.iloc[index]['perimeter_mean']
        area = self.df.iloc[index]['area_mean']
        
        # Simplified fractal dimension approximation: log(perimeter) / log(area)
        if area > 0:
            approx_fractal = np.log(perimeter) / np.log(area)
        else:
            approx_fractal = 0
        
        # Recursive call
        return self.recursive_fractal_dimension(index + 1, accumulator + approx_fractal)
    
    def compute_fractal_dimension(self):
        """
        Compute and compare recursive fractal dimension with actual values
        """
        print("\n" + "="*60)
        print("STEP 7: Fractal Dimension Approximation (Recursive)")
        print("="*60)
        
        # Get recursive approximation
        avg_approx, n_samples = self.recursive_fractal_dimension()
        
        # Get actual fractal dimension mean
        actual_mean = self.df['fractal_dimension_mean'].mean()
        
        print(f"Recursive approximation (average): {avg_approx:.6f}")
        print(f"Actual fractal_dimension_mean: {actual_mean:.6f}")
        print(f"Difference: {abs(avg_approx - actual_mean):.6f}")
        print(f"Processed {n_samples} samples recursively")
        
        # Compare first few samples
        print("\n📊 Comparison for first 5 samples:")
        for i in range(min(5, len(self.df))):
            perimeter = self.df.iloc[i]['perimeter_mean']
            area = self.df.iloc[i]['area_mean']
            approx = np.log(perimeter) / np.log(area) if area > 0 else 0
            actual = self.df.iloc[i]['fractal_dimension_mean']
            print(f"  Sample {i}: Approx={approx:.4f}, Actual={actual:.4f}")
    
    def comprehensive_statistics(self):
        """
        Step 8: Comprehensive statistics calculation
        """
        print("\n" + "="*60)
        print("STEP 8: Comprehensive Statistics")
        print("="*60)
        
        features = ['radius_mean', 'area_mean', 'compactness_mean', 'concavity_mean']
        
        # Manual statistics functions
        def manual_mean(data):
            return sum(data) / len(data) if data else 0
        
        def manual_median(data):
            sorted_data = sorted(data)
            n = len(sorted_data)
            if n % 2 == 0:
                return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
            return sorted_data[n//2]
        
        def manual_mode(data):
            counter = Counter(data)
            if not counter:
                return None
            max_count = max(counter.values())
            modes = [value for value, count in counter.items() if count == max_count]
            return modes[0] if len(modes) == 1 else modes
        
        print("\n📊 Manual Statistics Calculation:")
        for feature in features:
            data_list = self.df[feature].tolist()
            print(f"\n  {feature}:")
            print(f"    Mean: {manual_mean(data_list):.4f}")
            print(f"    Median: {manual_median(data_list):.4f}")
            print(f"    Mode: {manual_mode(data_list)}")
            print(f"    Min: {min(data_list):.4f}")
            print(f"    Max: {max(data_list):.4f}")
        
        # Pandas statistics for comparison
        print("\n📊 Pandas Statistics (for comparison):")
        print(self.df[features].describe())
        
        # Group statistics by diagnosis
        print("\n📊 Statistics Grouped by Diagnosis:")
        grouped_stats = {}
        for diagnosis in self.df['diagnosis'].unique():
            grouped_stats[diagnosis] = {}
            subset = self.df[self.df['diagnosis'] == diagnosis]
            
            for feature in features:
                grouped_stats[diagnosis][feature] = {
                    'mean': subset[feature].mean(),
                    'median': subset[feature].median(),
                    'std': subset[feature].std()
                }
            
            print(f"\n  Diagnosis: {diagnosis} (Malignant if 'M', Benign if 'B')")
            print(f"    Number of samples: {len(subset)}")
            for feature in features:
                stats = grouped_stats[diagnosis][feature]
                print(f"    {feature}: mean={stats['mean']:.2f}, "
                      f"median={stats['median']:.2f}, std={stats['std']:.2f}")
    
    def generate_report(self):
        """
        Step 9: Generate final analysis report
        """
        print("\n" + "="*60)
        print("STEP 9: Generating Final Report")
        print("="*60)
        
        report_filename = "breast_cancer_analysis_report.txt"
        
        try:
            with open(report_filename, 'w') as report_file:
                # Write report header
                report_file.write("="*80 + "\n")
                report_file.write("BREAST CANCER WISCONSIN ANALYSIS REPORT\n")
                report_file.write("="*80 + "\n\n")
                
                # Dataset overview
                report_file.write("1. DATASET OVERVIEW\n")
                report_file.write("-"*40 + "\n")
                report_file.write(f"Total samples: {len(self.df)}\n")
                report_file.write(f"Total features: {len(self.df.columns)}\n")
                report_file.write(f"Malignant cases (M): {len(self.df[self.df['diagnosis'] == 'M'])}\n")
                report_file.write(f"Benign cases (B): {len(self.df[self.df['diagnosis'] == 'B'])}\n\n")
                
                # Key findings
                report_file.write("2. KEY FINDINGS\n")
                report_file.write("-"*40 + "\n")
                
                # Malignancy risk score distribution
                risk_dist = self.df['MalignancyRiskScore'].value_counts().sort_index()
                report_file.write("Malignancy Risk Score Distribution:\n")
                for score, count in risk_dist.items():
                    report_file.write(f"  Score {score}: {count} samples\n")
                
                # Feature correlations with diagnosis
                report_file.write("\n3. FEATURE ANALYSIS\n")
                report_file.write("-"*40 + "\n")
                features = ['radius_mean', 'area_mean', 'concavity_mean']
                for feature in features:
                    m_mean = self.df[self.df['diagnosis'] == 'M'][feature].mean()
                    b_mean = self.df[self.df['diagnosis'] == 'B'][feature].mean()
                    report_file.write(f"{feature}:\n")
                    report_file.write(f"  Malignant mean: {m_mean:.2f}\n")
                    report_file.write(f"  Benign mean: {b_mean:.2f}\n")
                    report_file.write(f"  Difference: {abs(m_mean - b_mean):.2f}\n\n")
                
                # New column insights
                report_file.write("4. ENGINEERED FEATURES\n")
                report_file.write("-"*40 + "\n")
                report_file.write(f"Worst_Area_Mean range: {self.df['Worst_Area_Mean'].min():.2f} - "
                                f"{self.df['Worst_Area_Mean'].max():.2f}\n")
                report_file.write(f"Radius_Texture_Index range: {self.df['Radius_Texture_Index'].min():.2f} - "
                                f"{self.df['Radius_Texture_Index'].max():.2f}\n")
                
                # Recommendations
                report_file.write("\n5. RECOMMENDATIONS\n")
                report_file.write("-"*40 + "\n")
                report_file.write("Based on the analysis:\n")
                report_file.write("1. Features like radius_mean, area_mean, and concavity_mean show strong\n")
                report_file.write("   discriminatory power between malignant and benign cases.\n")
                report_file.write("2. The engineered MalignancyRiskScore can be used as a quick screening tool.\n")
                report_file.write("3. Further analysis with machine learning models is recommended for\n")
                report_file.write("   automated diagnosis support.\n")
            
            print(f"✅ Report successfully saved as '{report_filename}'")
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")
    
    def run_full_analysis(self):
        """
        Run all analysis steps
        """
        print("\n" + "🎯"*30)
        print("BREAST CANCER WISCONSIN STATISTICAL ANALYSIS")
        print("🎯"*30)
        
        # Step 1: Load data
        if not self.load_data():
            return
        
        # Step 2: Data exploration
        self.explore_data()
        
        # Step 3: Handle missing values
        self.handle_missing_values()
        
        # Step 4: Create new columns
        self.create_new_columns()
        
        # Step 5: Radius mean checks
        self.radius_mean_checks()
        
        # Step 6: Reduce calculations
        self.custom_reduce_calculations()
        
        # Step 7: Fractal dimension
        self.compute_fractal_dimension()
        
        # Step 8: Comprehensive statistics
        self.comprehensive_statistics()
        
        # Step 9: Generate report
        self.generate_report()
        
        print("\n" + "✅"*30)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("✅"*30)


# Main execution
if __name__ == "__main__":
    # Update this path to your actual CSV file location
    FILE_PATH = "data.csv"  # Change this to your actual file path
    
    # Create analyzer instance and run analysis
    analyzer = BreastCancerAnalyzer(FILE_PATH)
    analyzer.run_full_analysis()

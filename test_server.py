"""
Test script to verify server functionality
"""
import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.title("BiteBase Server Test")
    
    # Test basic Streamlit functionality
    st.write("Testing basic functionality...")
    
    # Test data display
    data = pd.DataFrame({
        'test_col': np.random.randn(5),
        'test_col2': np.random.randn(5)
    })
    st.write("Test DataFrame:")
    st.dataframe(data)
    
    # Test interactive elements
    st.sidebar.header("Test Controls")
    test_slider = st.sidebar.slider("Test Slider", 0, 100, 50)
    st.write(f"Slider value: {test_slider}")
    
    # Test plotting
    st.subheader("Test Plot")
    st.line_chart(data)
    
    st.success("Server test completed successfully!")

if __name__ == "__main__":
    main()
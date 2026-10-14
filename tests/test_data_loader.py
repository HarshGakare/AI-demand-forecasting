import pandas as pd
import pytest

from src.data_loader import (
    load_raw_data,
    validate_columns,
    merge_store_data
)

def test_load_raw_data():
    train, test, store, sample = load_raw_data()
    
    assert not train.empty
    assert not test.empty
    assert not store.empty
    assert not sample.empty
    

def test_expected_columns():
    train, test, store, _ = load_raw_data()
    
    validate_columns(train,test,store)
    

def test_store_merge():
    train, test, store, _ = load_raw_data()
    
    train_merged, test_merged = merge_store_data(
        train, test, store
    )
    
    assert len(train_merged) == len(train)
    assert len(test_merged) == len(test)
    
    assert "StoreType" in train_merged.columns
    assert "Assortment" in test_merged.columns
    

def test_data_column():
    train,_,_,_ = load_raw_data()
    
    assert (train["Sales"] >= 0).all()
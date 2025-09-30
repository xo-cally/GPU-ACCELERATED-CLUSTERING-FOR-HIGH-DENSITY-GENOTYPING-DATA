// src/lib.rs
use std::collections::HashMap;
use std::io;

// Data structures for genotyping data
#[derive(Debug, Clone)]
pub struct IntensityPair {
    pub green: f64,  // Allele A intensity
    pub red: f64,    // Allele B intensity
}

#[derive(Debug, Clone)]
pub struct SNPData {
    pub id: String,
    pub chromosome: String,
    pub position: u64,
    pub intensities: Vec<IntensityPair>,  // One per individual
}

#[derive(Debug, Clone)]
pub struct Individual {
    pub id: String,
    pub sample_name: String,
}

#[derive(Debug)]
pub struct GenotypeDataset {
    pub individuals: Vec<Individual>,
    pub snps: Vec<SNPData>,
    pub individual_lookup: HashMap<String, usize>,
    pub snp_lookup: HashMap<String, usize>,
}

// Error handling for the pipeline
#[derive(Debug)]
pub enum PipelineError {
    IoError(io::Error),
    ParseError(String),
    FormatError(String),
}

impl From<io::Error> for PipelineError {
    fn from(error: io::Error) -> Self {
        PipelineError::IoError(error)
    }
}

impl GenotypeDataset {
    pub fn new() -> Self {
        GenotypeDataset {
            individuals: Vec::new(),
            snps: Vec::new(),
            individual_lookup: HashMap::new(),
            snp_lookup: HashMap::new(),
        }
    }

    // Add an individual to the dataset
    pub fn add_individual(&mut self, individual: Individual) -> usize {
        let index = self.individuals.len();
        self.individual_lookup.insert(individual.id.clone(), index);
        self.individuals.push(individual);
        index
    }

    // Add a SNP to the dataset
    pub fn add_snp(&mut self, snp: SNPData) -> usize {
        let index = self.snps.len();
        self.snp_lookup.insert(snp.id.clone(), index);
        self.snps.push(snp);
        index
    }

    // Get dataset dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.individuals.len(), self.snps.len())
    }
}


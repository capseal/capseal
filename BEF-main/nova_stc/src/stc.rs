use std::sync::Arc;

use nova_snark::frontend::gadgets::num::AllocatedNum;
use nova_snark::frontend::gadgets::poseidon::{
    Elt, IOPattern, PoseidonConstants, Simplex, Sponge, SpongeAPI, SpongeCircuit, SpongeOp,
    SpongeTrait,
};
use nova_snark::frontend::{ConstraintSystem, SynthesisError};
use nova_snark::traits::circuit::StepCircuit;
use nova_snark::traits::PrimeFieldExt;
use typenum::{U3, U8};

pub type ChunkArity = U8;
pub type RootArity = U3;

#[derive(Clone, Debug, PartialEq)]
pub struct StcState<F> {
    pub n: F,
    pub root: F,
    pub s: Vec<F>,
    pub pow: Vec<F>,
}

impl<F: PrimeFieldExt> StcState<F> {
    pub fn initial(challenges: usize) -> Self {
        Self {
            n: F::ZERO,
            root: F::ZERO,
            s: vec![F::ZERO; challenges],
            pow: vec![F::ONE; challenges],
        }
    }

    pub fn to_vec(&self) -> Vec<F> {
        let mut out = Vec::with_capacity(2 + 2 * self.s.len());
        out.push(self.n);
        out.push(self.root);
        out.extend_from_slice(&self.s);
        out.extend_from_slice(&self.pow);
        out
    }

    pub fn from_vec(values: &[F]) -> Option<Self> {
        if values.len() < 2 || (values.len() - 2) % 2 != 0 {
            return None;
        }
        let m = (values.len() - 2) / 2;
        Some(Self {
            n: values[0],
            root: values[1],
            s: values[2..2 + m].to_vec(),
            pow: values[2 + m..].to_vec(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct StcChunk<F> {
    pub values: Vec<F>,
}

impl<F> StcChunk<F> {
    pub fn new(values: Vec<F>) -> Self {
        Self { values }
    }
}

#[derive(Clone, Debug)]
pub struct StcParams<F: PrimeFieldExt> {
    pub challenges: Vec<F>,
    pub chunk_len: usize,
    pub poseidon_chunk: PoseidonConstants<F, ChunkArity>,
    pub poseidon_root: PoseidonConstants<F, RootArity>,
}

impl<F: PrimeFieldExt> StcParams<F> {
    pub fn new(challenges: Vec<F>, chunk_len: usize) -> Self {
        Self {
            challenges,
            chunk_len,
            poseidon_chunk: PoseidonConstants::new(),
            poseidon_root: PoseidonConstants::new(),
        }
    }

    pub fn challenges_len(&self) -> usize {
        self.challenges.len()
    }
}

fn hash_chunk_native<F: PrimeFieldExt>(
    inputs: &[F],
    constants: &PoseidonConstants<F, ChunkArity>,
) -> F {
    let mut sponge = Sponge::<F, ChunkArity>::new_with_constants(constants, Simplex);
    let acc = &mut ();
    let pattern = IOPattern(vec![
        SpongeOp::Absorb(inputs.len() as u32),
        SpongeOp::Squeeze(1),
    ]);
    sponge.start(pattern, None, acc);
    SpongeAPI::absorb(&mut sponge, inputs.len() as u32, inputs, acc);
    let out = SpongeAPI::squeeze(&mut sponge, 1, acc);
    sponge.finish(acc).expect("poseidon sponge finish");
    out[0]
}

fn hash_root_native<F: PrimeFieldExt>(
    inputs: &[F],
    constants: &PoseidonConstants<F, RootArity>,
) -> F {
    let mut sponge = Sponge::<F, RootArity>::new_with_constants(constants, Simplex);
    let acc = &mut ();
    let pattern = IOPattern(vec![
        SpongeOp::Absorb(inputs.len() as u32),
        SpongeOp::Squeeze(1),
    ]);
    sponge.start(pattern, None, acc);
    SpongeAPI::absorb(&mut sponge, inputs.len() as u32, inputs, acc);
    let out = SpongeAPI::squeeze(&mut sponge, 1, acc);
    sponge.finish(acc).expect("poseidon sponge finish");
    out[0]
}

fn hash_chunk_circuit<F: PrimeFieldExt, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    inputs: &[AllocatedNum<F>],
    constants: &PoseidonConstants<F, ChunkArity>,
) -> Result<AllocatedNum<F>, SynthesisError> {
    let mut ns = cs.namespace(|| "chunk_poseidon");
    let output = {
        let mut sponge = SpongeCircuit::<F, ChunkArity, _>::new_with_constants(constants, Simplex);
        let acc = &mut ns;
        let pattern = IOPattern(vec![
            SpongeOp::Absorb(inputs.len() as u32),
            SpongeOp::Squeeze(1),
        ]);
        sponge.start(pattern, None, acc);
        let elts: Vec<Elt<F>> = inputs
            .iter()
            .map(|value| Elt::Allocated(value.clone()))
            .collect();
        SpongeAPI::absorb(&mut sponge, elts.len() as u32, &elts, acc);
        let squeezed = SpongeAPI::squeeze(&mut sponge, 1, acc);
        sponge
            .finish(acc)
            .map_err(|_| SynthesisError::Unsatisfiable)?;
        squeezed
    };
    let digest = Elt::ensure_allocated(
        &output[0],
        &mut ns.namespace(|| "chunk_poseidon_output"),
        true,
    )?;
    Ok(digest)
}

fn hash_root_circuit<F: PrimeFieldExt, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    inputs: &[AllocatedNum<F>],
    constants: &PoseidonConstants<F, RootArity>,
) -> Result<AllocatedNum<F>, SynthesisError> {
    let mut ns = cs.namespace(|| "root_poseidon");
    let output = {
        let mut sponge = SpongeCircuit::<F, RootArity, _>::new_with_constants(constants, Simplex);
        let acc = &mut ns;
        let pattern = IOPattern(vec![
            SpongeOp::Absorb(inputs.len() as u32),
            SpongeOp::Squeeze(1),
        ]);
        sponge.start(pattern, None, acc);
        let elts: Vec<Elt<F>> = inputs
            .iter()
            .map(|value| Elt::Allocated(value.clone()))
            .collect();
        SpongeAPI::absorb(&mut sponge, elts.len() as u32, &elts, acc);
        let squeezed = SpongeAPI::squeeze(&mut sponge, 1, acc);
        sponge
            .finish(acc)
            .map_err(|_| SynthesisError::Unsatisfiable)?;
        squeezed
    };
    let digest = Elt::ensure_allocated(
        &output[0],
        &mut ns.namespace(|| "root_poseidon_output"),
        true,
    )?;
    Ok(digest)
}

pub fn update_state<F: PrimeFieldExt>(
    params: &StcParams<F>,
    prev: &StcState<F>,
    chunk: &StcChunk<F>,
) -> StcState<F> {
    let chunk_digest = hash_chunk_native(&chunk.values, &params.poseidon_chunk);
    let root_inputs = [prev.root, chunk_digest, prev.n];
    let root_next = hash_root_native(&root_inputs, &params.poseidon_root);

    let mut s_next = Vec::with_capacity(params.challenges_len());
    let mut pow_next = Vec::with_capacity(params.challenges_len());

    for (j, challenge) in params.challenges.iter().enumerate() {
        let mut s_val = prev.s[j];
        let mut pow_val = prev.pow[j];
        for value in &chunk.values {
            s_val += *value * pow_val;
            pow_val *= *challenge;
        }
        s_next.push(s_val);
        pow_next.push(pow_val);
    }

    StcState {
        n: prev.n + F::from(chunk.values.len() as u64),
        root: root_next,
        s: s_next,
        pow: pow_next,
    }
}

#[derive(Clone)]
pub struct StcStepCircuit<F: PrimeFieldExt> {
    params: Arc<StcParams<F>>,
    chunk: StcChunk<F>,
}

impl<F: PrimeFieldExt> StcStepCircuit<F> {
    pub fn new(params: Arc<StcParams<F>>, chunk: StcChunk<F>) -> Self {
        if chunk.values.len() != params.chunk_len {
            panic!(
                "chunk length mismatch: expected {} got {}",
                params.chunk_len,
                chunk.values.len()
            );
        }
        Self { params, chunk }
    }

    pub fn blank(params: Arc<StcParams<F>>) -> Self {
        let zero_chunk = StcChunk::new(vec![F::ZERO; params.chunk_len]);
        Self::new(params, zero_chunk)
    }
}

impl<F: PrimeFieldExt> StepCircuit<F> for StcStepCircuit<F> {
    fn arity(&self) -> usize {
        2 + 2 * self.params.challenges_len()
    }

    fn synthesize<CS>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError>
    where
        CS: ConstraintSystem<F>,
    {
        let m = self.params.challenges_len();
        if z.len() != self.arity() {
            return Err(SynthesisError::Unsatisfiable);
        }

        let n_prev = z[0].clone();
        let root_prev = z[1].clone();
        let s_prev = z[2..2 + m].to_vec();
        let pow_prev = z[2 + m..].to_vec();

        let chunk_vars = self
            .chunk
            .values
            .iter()
            .enumerate()
            .map(|(i, value)| {
                AllocatedNum::alloc(cs.namespace(|| format!("chunk_{i}")), || Ok(*value))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let chunk_len_scalar = F::from(chunk_vars.len() as u64);

        let chunk_digest = hash_chunk_circuit(cs, &chunk_vars, &self.params.poseidon_chunk)?;

        let root_inputs = vec![root_prev.clone(), chunk_digest.clone(), n_prev.clone()];
        let root_next = hash_root_circuit(cs, &root_inputs, &self.params.poseidon_root)?;

        let n_next = AllocatedNum::alloc(cs.namespace(|| "n_next"), || {
            let mut val = n_prev
                .get_value()
                .ok_or(SynthesisError::AssignmentMissing)?;
            val += chunk_len_scalar;
            Ok(val)
        })?;
        cs.enforce(
            || "n update",
            |lc| lc + n_prev.get_variable() + (chunk_len_scalar, CS::one()),
            |lc| lc + CS::one(),
            |lc| lc + n_next.get_variable(),
        );

        let mut outputs = Vec::with_capacity(self.arity());
        let mut s_outputs = Vec::with_capacity(m);
        let mut pow_outputs = Vec::with_capacity(m);
        outputs.push(n_next);
        outputs.push(root_next);

        for (j, challenge) in self.params.challenges.iter().enumerate() {
            let mut s_acc = s_prev[j].clone();
            let mut pow_acc = pow_prev[j].clone();

            for (k, chunk_var) in chunk_vars.iter().enumerate() {
                let prod = chunk_var.mul(cs.namespace(|| format!("s_mul_{j}_{k}")), &pow_acc)?;
                s_acc = s_acc.add(cs.namespace(|| format!("s_acc_{j}_{k}")), &prod)?;

                let pow_next =
                    AllocatedNum::alloc(cs.namespace(|| format!("pow_next_{j}_{k}")), || {
                        let mut val = pow_acc
                            .get_value()
                            .ok_or(SynthesisError::AssignmentMissing)?;
                        val *= *challenge;
                        Ok(val)
                    })?;
                cs.enforce(
                    || format!("pow_update_{j}_{k}"),
                    |lc| lc + pow_acc.get_variable(),
                    |lc| lc + (*challenge, CS::one()),
                    |lc| lc + pow_next.get_variable(),
                );
                pow_acc = pow_next;
            }

            s_outputs.push(s_acc);
            pow_outputs.push(pow_acc);
        }

        outputs.extend(s_outputs);
        outputs.extend(pow_outputs);

        Ok(outputs)
    }
}

use crate::jaxpr::{Jaxpr, JaxprEqn, Atom};
use crate::primitive::Primitive;

fn primitive_to_stablehlo(prim: &Primitive) -> &'static str {
    match prim {
        Primitive::Add => "stablehlo.add",
        Primitive::Mul => "stablehlo.multiply",
        Primitive::Neg => "stablehlo.negate",
        Primitive::Sin => "stablehlo.sine",
        Primitive::Cos => "stablehlo.cosine",
        Primitive::Greater => "stablehlo.compare",
        Primitive::Less => "stablehlo.compare",
        Primitive::ReduceSum => "stablehlo.reduce",
    }
}

fn atom_to_mlir(atom: &Atom) -> String {
    match atom {
        Atom::Var(v) => format!("%{}", v.display_name()),
        Atom::Lit(f) => format!("%cst_{}", f.to_bits()),
    }
}

fn lower_eqn(eqn: &JaxprEqn) -> String {
    let op = primitive_to_stablehlo(&eqn.primitive);
    let inputs: Vec<String> = eqn.inputs.iter().map(atom_to_mlir).collect();
    let output = format!("%{}", eqn.outputs[0].display_name());
    
    match eqn.primitive {
        Primitive::Neg | Primitive::Sin | Primitive::Cos => {
            format!("    {} = {} {} : tensor<f64>", output, op, inputs[0])
        }
        Primitive::Add | Primitive::Mul => {
            format!("    {} = {} {}, {} : tensor<f64>", output, op, inputs[0], inputs[1])
        }
        Primitive::Greater => {
            format!("    {} = stablehlo.compare GT, {}, {} : (tensor<f64>, tensor<f64>) -> tensor<i1>", 
                    output, inputs[0], inputs[1])
        }
        Primitive::Less => {
            format!("    {} = stablehlo.compare LT, {}, {} : (tensor<f64>, tensor<f64>) -> tensor<i1>", 
                    output, inputs[0], inputs[1])
        }
        Primitive::ReduceSum => {
            format!("    {} = \"stablehlo.reduce\"({}) : (tensor<f64>) -> tensor<f64>", 
                    output, inputs.join(", "))
        }
    }
}

pub fn lower_to_stablehlo(jaxpr: &Jaxpr) -> String {
    let mut lines = Vec::new();
    
    lines.push("module {".to_string());
    
    let in_args: Vec<String> = jaxpr.in_binders.iter()
        .map(|v| format!("%{}: tensor<f64>", v.display_name()))
        .collect();
    lines.push(format!("  func.func @main({}) -> tensor<f64> {{", in_args.join(", ")));
    
    let mut constants = Vec::new();
    for eqn in &jaxpr.eqns {
        for atom in &eqn.inputs {
            if let Atom::Lit(f) = atom {
                let cst_name = format!("%cst_{}", f.to_bits());
                let cst_def = format!("    {} = stablehlo.constant dense<{:.6}> : tensor<f64>", cst_name, f);
                if !constants.contains(&cst_def) {
                    constants.push(cst_def);
                }
            }
        }
    }
    for cst in constants {
        lines.push(cst);
    }
    
    for eqn in &jaxpr.eqns {
        lines.push(lower_eqn(eqn));
    }
    
    let out_atom = &jaxpr.out_atoms[0];
    let out_val = atom_to_mlir(out_atom);
    lines.push(format!("    return {} : tensor<f64>", out_val));
    
    lines.push("  }".to_string());
    lines.push("}".to_string());
    
    lines.join("\n")
}

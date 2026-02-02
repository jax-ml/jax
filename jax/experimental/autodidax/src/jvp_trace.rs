use crate::primitive::Primitive;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum JvpValue {
    Concrete(f64),
    Tracer(Arc<JvpTracer>),
}

#[derive(Debug, Clone)]
pub struct JvpTracer {
    pub primal: JvpValue,
    pub tangent: JvpValue,
}

impl JvpTracer {
    pub fn new(primal: JvpValue, tangent: JvpValue) -> Self {
        JvpTracer { primal, tangent }
    }

    pub fn from_floats(primal: f64, tangent: f64) -> Self {
        JvpTracer {
            primal: JvpValue::Concrete(primal),
            tangent: JvpValue::Concrete(tangent),
        }
    }
}

impl JvpValue {
    pub fn from_f64(val: f64) -> Self {
        JvpValue::Concrete(val)
    }

    pub fn tracer(primal: JvpValue, tangent: JvpValue) -> Self {
        JvpValue::Tracer(Arc::new(JvpTracer::new(primal, tangent)))
    }

    pub fn zeros_like(&self) -> Self {
        match self {
            JvpValue::Concrete(_) => JvpValue::Concrete(0.0),
            JvpValue::Tracer(t) => JvpValue::tracer(
                t.primal.zeros_like(),
                t.tangent.zeros_like(),
            ),
        }
    }

    pub fn get_concrete(&self) -> f64 {
        match self {
            JvpValue::Concrete(f) => *f,
            JvpValue::Tracer(t) => t.primal.get_concrete(),
        }
    }

    pub fn is_tracer(&self) -> bool {
        matches!(self, JvpValue::Tracer(_))
    }

    pub fn primal(&self) -> JvpValue {
        match self {
            JvpValue::Concrete(f) => JvpValue::Concrete(*f),
            JvpValue::Tracer(t) => t.primal.clone(),
        }
    }

    pub fn tangent(&self) -> JvpValue {
        match self {
            JvpValue::Concrete(_) => JvpValue::Concrete(0.0),
            JvpValue::Tracer(t) => t.tangent.clone(),
        }
    }
}

fn ensure_tracer(val: JvpValue) -> JvpValue {
    match val {
        JvpValue::Tracer(_) => val,
        JvpValue::Concrete(f) => JvpValue::tracer(
            JvpValue::Concrete(f),
            JvpValue::Concrete(0.0),
        ),
    }
}

pub fn jvp_neg(x: JvpValue) -> JvpValue {
    let x = ensure_tracer(x);
    match x {
        JvpValue::Tracer(t) => {
            let primal_out = match &t.primal {
                JvpValue::Tracer(_) => jvp_neg(t.primal.clone()),
                JvpValue::Concrete(f) => JvpValue::Concrete(-f),
            };
            let tangent_out = match &t.tangent {
                JvpValue::Tracer(_) => jvp_neg(t.tangent.clone()),
                JvpValue::Concrete(f) => JvpValue::Concrete(-f),
            };
            JvpValue::tracer(primal_out, tangent_out)
        }
        JvpValue::Concrete(f) => JvpValue::Concrete(-f),
    }
}

pub fn jvp_add(x: JvpValue, y: JvpValue) -> JvpValue {
    let x = ensure_tracer(x);
    let y = ensure_tracer(y);
    
    match (&x, &y) {
        (JvpValue::Tracer(tx), JvpValue::Tracer(ty)) => {
            let primal_out = if tx.primal.is_tracer() || ty.primal.is_tracer() {
                jvp_add(tx.primal.clone(), ty.primal.clone())
            } else {
                JvpValue::Concrete(tx.primal.get_concrete() + ty.primal.get_concrete())
            };
            let tangent_out = if tx.tangent.is_tracer() || ty.tangent.is_tracer() {
                jvp_add(tx.tangent.clone(), ty.tangent.clone())
            } else {
                JvpValue::Concrete(tx.tangent.get_concrete() + ty.tangent.get_concrete())
            };
            JvpValue::tracer(primal_out, tangent_out)
        }
        _ => unreachable!("ensure_tracer should have wrapped these"),
    }
}

pub fn jvp_mul(x: JvpValue, y: JvpValue) -> JvpValue {
    let x = ensure_tracer(x);
    let y = ensure_tracer(y);
    
    match (&x, &y) {
        (JvpValue::Tracer(tx), JvpValue::Tracer(ty)) => {
            let primal_out = if tx.primal.is_tracer() || ty.primal.is_tracer() {
                jvp_mul(tx.primal.clone(), ty.primal.clone())
            } else {
                JvpValue::Concrete(tx.primal.get_concrete() * ty.primal.get_concrete())
            };
            let has_nested = tx.primal.is_tracer() || ty.tangent.is_tracer() ||
                             tx.tangent.is_tracer() || ty.primal.is_tracer();
            let tangent_out = if has_nested {
                let term1 = jvp_mul(tx.primal.clone(), ty.tangent.clone());
                let term2 = jvp_mul(tx.tangent.clone(), ty.primal.clone());
                jvp_add(term1, term2)
            } else {
                let xp = tx.primal.get_concrete();
                let yt = ty.tangent.get_concrete();
                let xt = tx.tangent.get_concrete();
                let yp = ty.primal.get_concrete();
                JvpValue::Concrete(xp * yt + xt * yp)
            };
            JvpValue::tracer(primal_out, tangent_out)
        }
        _ => unreachable!("ensure_tracer should have wrapped these"),
    }
}

pub fn jvp_sin(x: JvpValue) -> JvpValue {
    let x = ensure_tracer(x);
    match x {
        JvpValue::Tracer(t) => {
            let (primal_out, tangent_out) = if t.primal.is_tracer() {
                let sin_primal = jvp_sin(t.primal.clone());
                let cos_primal = jvp_cos(t.primal.clone());
                let tangent = jvp_mul(cos_primal, t.tangent.clone());
                (sin_primal, tangent)
            } else {
                let p = t.primal.get_concrete();
                let primal = JvpValue::Concrete(p.sin());
                let cos_val = p.cos();
                let tangent = if t.tangent.is_tracer() {
                    jvp_mul(JvpValue::Concrete(cos_val), t.tangent.clone())
                } else {
                    JvpValue::Concrete(cos_val * t.tangent.get_concrete())
                };
                (primal, tangent)
            };
            JvpValue::tracer(primal_out, tangent_out)
        }
        JvpValue::Concrete(f) => JvpValue::Concrete(f.sin()),
    }
}

pub fn jvp_cos(x: JvpValue) -> JvpValue {
    let x = ensure_tracer(x);
    match x {
        JvpValue::Tracer(t) => {
            let (primal_out, tangent_out) = if t.primal.is_tracer() {
                let cos_primal = jvp_cos(t.primal.clone());
                let sin_primal = jvp_sin(t.primal.clone());
                let neg_sin = jvp_neg(sin_primal);
                let tangent = jvp_mul(neg_sin, t.tangent.clone());
                (cos_primal, tangent)
            } else {
                let p = t.primal.get_concrete();
                let primal = JvpValue::Concrete(p.cos());
                let neg_sin_val = -p.sin();
                let tangent = if t.tangent.is_tracer() {
                    jvp_mul(JvpValue::Concrete(neg_sin_val), t.tangent.clone())
                } else {
                    JvpValue::Concrete(neg_sin_val * t.tangent.get_concrete())
                };
                (primal, tangent)
            };
            JvpValue::tracer(primal_out, tangent_out)
        }
        JvpValue::Concrete(f) => JvpValue::Concrete(f.cos()),
    }
}

pub fn jvp_primitive(prim: Primitive, args: Vec<JvpValue>) -> JvpValue {
    match prim {
        Primitive::Add => {
            assert_eq!(args.len(), 2);
            jvp_add(args[0].clone(), args[1].clone())
        }
        Primitive::Mul => {
            assert_eq!(args.len(), 2);
            jvp_mul(args[0].clone(), args[1].clone())
        }
        Primitive::Neg => {
            assert_eq!(args.len(), 1);
            jvp_neg(args[0].clone())
        }
        Primitive::Sin => {
            assert_eq!(args.len(), 1);
            jvp_sin(args[0].clone())
        }
        Primitive::Cos => {
            assert_eq!(args.len(), 1);
            jvp_cos(args[0].clone())
        }
        Primitive::Greater => {
            assert_eq!(args.len(), 2);
            let x = args[0].get_concrete();
            let y = args[1].get_concrete();
            let val = if x > y { 1.0 } else { 0.0 };
            JvpValue::tracer(JvpValue::Concrete(val), JvpValue::Concrete(0.0))
        }
        Primitive::Less => {
            assert_eq!(args.len(), 2);
            let x = args[0].get_concrete();
            let y = args[1].get_concrete();
            let val = if x < y { 1.0 } else { 0.0 };
            JvpValue::tracer(JvpValue::Concrete(val), JvpValue::Concrete(0.0))
        }
        Primitive::ReduceSum => {
            let mut result = JvpValue::Concrete(0.0);
            for arg in args {
                result = jvp_add(result, arg);
            }
            result
        }
    }
}

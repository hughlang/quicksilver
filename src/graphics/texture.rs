/// Model object that holds GL instructions and configs.
///
///
use crate::{
    backend::Backend,
    graphics::{GpuTriangle, Vertex},
    lifecycle::Window,
};

/// This is a wrapper object that holds the info necessary for executing a set of GL instructions.
/// This modularity allows Quicksilver to handle multiple GL processing jobs separately with
/// different shader programs and expected outputs.
// #[derive(Clone)]
pub struct GLTexture {
    /// List of name and field width values
    pub fields: Vec<(String, u32)>,
    /// Function that serializes a Vertex struct into the vec of f32 vals that the GPU shader expects.
    pub texture_id: u32,
    /// The id returned by glCreateProgram in backend.link_program.
    pub program_id: u32,
    /// The id returned when glCreateShader in backend.compile_shader for the vertex shader
    vertex_id: u32,
    /// The id returned when glCreateShader in backend.compile_shader for the fragment shader
    fragment_id: u32,
    pub serializer: Box<dyn Fn(Vertex) -> Vec<f32> + 'static>,
    /// The ImageData.id value returned by glGenTextures in backend.create_texture.
}

impl Default for GLTexture {
    fn default() -> Self {
        let default = |_vertex| -> Vec<f32> {
            Vec::new()
        };
        GLTexture {
            fields: Vec::new(),
            texture_id: 0,
            program_id: 0,
            vertex_id: 0,
            fragment_id: 0,
            serializer: Box::new(default),
        }
    }
}

impl GLTexture {

    /// Initialize both shaders using the provided strings which contain OpenGL/WebGL code
    pub fn init_shaders(mut self, vertex_shader: &str, fragment_shader: &str, window: &mut Window) -> Self {
        unsafe {
            let vs = window.backend().compile_shader(vertex_shader, gl::VERTEX_SHADER);
            let fs = window.backend().compile_shader(fragment_shader, gl::FRAGMENT_SHADER);
            if vs.is_ok() && fs.is_ok() {
                let vs = vs.unwrap();
                let fs = fs.unwrap();
                self.vertex_id = vs;
                self.fragment_id = fs;
                let pid = window.backend().link_program(vs, fs);
                if pid.is_ok() {
                    self.program_id = pid.unwrap();
                }
            }
        }
        self
    }
    /// Set the fields that match the shader program inputs
    pub fn with_fields(mut self, fields: &[(&str, u32)], out_color: &str, window: &mut Window) -> Self {
        self.fields = fields.iter().map(|(s, n)| (s.to_string(), n.clone())).collect();
        self
    }

    /// Set the closure function that is used to convert a vertex into a vector of u32 values
    /// that match the data expected by the GL vertex shader
    pub fn set_serializer<C>(&mut self, cb: C)
    where C: Fn(Vertex) -> Vec<f32> + 'static,
    {
        self.serializer = Box::new(cb);
    }
}

// #[cfg(not(target_arch="wasm32"))]
// impl Drop for GLTexture {
//     fn drop(&mut self) {
//         unsafe {
//             gl::DeleteProgram(self.program_id);
//             gl::DeleteShader(self.fragment_id);
//             gl::DeleteShader(self.vertex_id);
//         }
//     }
// }
// #[cfg(target_arch="wasm32")]
// impl Drop for GLTexture {
//     fn drop(&mut self) {
//         self.gl_ctx.delete_program(Some(&self.program_id));
//         self.gl_ctx.delete_shader(Some(&self.fragment_id));
//         self.gl_ctx.delete_shader(Some(&self.vertex_id));
//     }
// }

pub struct DrawTask {
    /// All the vertices in the task
    pub vertices: Vec<Vertex>,
    /// All the triangles in the task
    pub triangles: Vec<GpuTriangle>,

}

impl DrawTask {
    pub fn new() -> Self {
        DrawTask {
            vertices: Vec::new(),
            triangles: Vec::new(),
        }
    }
}
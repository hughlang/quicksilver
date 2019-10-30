use crate::{
    Result,
    backend::{Backend, ImageData, SurfaceData, VERTEX_SIZE},
    error::QuicksilverError,
    geom::{Rectangle, Vector},
    graphics::{BlendMode, Color, MeshTask, GpuTriangle, Image, ImageScaleStrategy, PixelFormat, Surface, Texture, Vertex},
    input::MouseCursor
};
use gl::types::*;
use glutin::{
    WindowedContext, dpi::LogicalSize,
};
use std::{
    ffi::{CStr, CString},
    mem::size_of,
    ops::Range,
    os::raw::c_void,
    ptr::{self, null as nullptr},
    str,
};

// *****************************************************************************************************
// OpenGL debug functions
// *****************************************************************************************************

/// This function call doesn't work on macOS
extern "system" fn debug_output_gl(_source: GLenum, _type: GLenum, _id: GLuint, _severity: GLenum,
    _length: GLsizei, message: *const GLchar, _param: *mut GLvoid) {
    unsafe {
        let slice = {
            assert!(!message.is_null());
            CStr::from_ptr(message)
        };
        let text = slice.to_str().unwrap();
        log::debug!("OpenGL Debug: {}", text);
    }
}

pub struct TextureUnit {
    /// The id returned by glCreateProgram in backend.link_program.
    pub program_id: u32,
    /// The id returned when glCreateShader in backend.compile_shader for the vertex shader
    pub vertex_id: u32,
    /// The id returned when glCreateShader in backend.compile_shader for the fragment shader
    pub fragment_id: u32,
    /// The id value returned by glGenTextures
    pub texture_id: u32,
    /// The location
    pub location_id: i32,
    /// The serializer function
    pub serializer: Box<dyn Fn(Vertex) -> Vec<f32> + 'static>,
}

// *****************************************************************************************************
// GL3Backend
// *****************************************************************************************************

pub struct GL3Backend {
    context: WindowedContext,
    texture: u32,
    vertices: Vec<f32>,
    indices: Vec<u32>,
    vertex_length: usize,
    index_length: usize,
    shader: u32,
    fragment: u32,
    vertex: u32,
    vbo: u32,
    ebo: u32,
    vao: u32,
    texture_location: i32,
    texture_mode: u32,
    tex_units: Vec<TextureUnit>,
}

const NULL_TEXTURE_ID: u32 = 0;

fn format_gl(format: PixelFormat) -> u32 {
    match format {
        PixelFormat::Alpha => gl::RED,
        PixelFormat::RGB => gl::RGB,
        PixelFormat::RGBA => gl::RGBA
    }
}

fn byte_size(format: PixelFormat) -> u32 {
    match format {
        PixelFormat::RGBA => 4,
        PixelFormat::RGB => 3,
        PixelFormat::Alpha => 1,
    }
}

impl Backend for GL3Backend {
    type Platform = WindowedContext;

    unsafe fn new(context: WindowedContext, texture_mode: ImageScaleStrategy, multisample: bool) -> Result<GL3Backend> {
        if gl::DebugMessageCallback::is_loaded() {
            gl::Enable(gl::DEBUG_OUTPUT);
            gl::DebugMessageCallback(debug_output_gl as GLDEBUGPROC, ptr::null());
            gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS);
        }
        let texture_mode = match texture_mode {
            ImageScaleStrategy::Pixelate => gl::NEAREST,
            ImageScaleStrategy::Blur => gl::LINEAR
        };
        let vao = {
            let mut array = 0;
            gl::GenVertexArrays(1, &mut array as *mut u32);
            array
        };
        let raw = gl::GetString(gl::VERSION);
        let version = String::from_utf8(CStr::from_ptr(raw as *const _).to_bytes().to_vec()).unwrap();
        println!(">>> OpenGL version={:?}", version);

        gl::BindVertexArray(vao);
        let mut buffers = [0, 0];
        gl::GenBuffers(2, (&mut buffers).as_mut_ptr());
        let [vbo, ebo] = buffers;
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
        gl::BlendFuncSeparate(
            gl::SRC_ALPHA,
            gl::ONE_MINUS_SRC_ALPHA,
            gl::ONE,
            gl::ONE_MINUS_SRC_ALPHA,
        );
        gl::Enable(gl::BLEND);
        // gl::Enable(gl::FRAMEBUFFER_SRGB);
        // gl::ClearColor(0.02, 0.02, 0.02, 1.0);

        if multisample {
            gl::Enable(gl::MULTISAMPLE);
        }

        let shader: u32 = 0;
        let fragment: u32 = 0;
        let vertex:u32 = 0;

        log::debug!("### GL3Backend.new – Using program={:?} vao={} vbo={} ebo={}", shader, vao, vbo, ebo);
        let mut backend = GL3Backend {
            context,
            texture: NULL_TEXTURE_ID,
            vertices: Vec::with_capacity(1024),
            indices: Vec::with_capacity(1024),
            vertex_length: 0,
            index_length: 0,
            shader, fragment, vertex,
            vbo, ebo, vao,
            texture_location: 0,
            texture_mode,
            tex_units: Vec::new(),
        };

        // Create the default texture which is shared by all of the standard Drawables when
        // DrawElements is called. The texture_idx will be 0
        let texture = Texture::new("default")
            .with_shaders(DEFAULT_VERTEX_SHADER, DEFAULT_FRAGMENT_SHADER)
            .with_fields(TEX_FIELDS, serialize_vertex, OUT_COLOR, SAMPLER);

        let texture_idx = backend.create_texture_unit(&texture)?;
        let unit = &backend.tex_units[texture_idx];
        backend.shader = unit.program_id;
        log::debug!("### Created default texture_unit idx={} texture_id={} program_id={}", texture_idx, unit.texture_id, unit.program_id);
        // let result = backend.upload_texture(0, &[], 1024, 1024, PixelFormat::RGBA);


        Ok(backend)
    }

    unsafe fn clear(&mut self, col: Color) {
        gl::ClearColor(col.r, col.g, col.b, col.a);
        gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
    }

    unsafe fn set_blend_mode(&mut self, blend: BlendMode) {
        gl::BlendFunc(gl::ONE, gl::ONE);
        gl::BlendEquationSeparate(blend as u32, gl::FUNC_ADD);
    }

    unsafe fn reset_blend_mode(&mut self) {
        gl::BlendFuncSeparate(
            gl::SRC_ALPHA,
            gl::ONE_MINUS_SRC_ALPHA,
            gl::ONE,
            gl::ONE_MINUS_SRC_ALPHA,
        );
        gl::BlendEquationSeparate(gl::FUNC_ADD, gl::FUNC_ADD);
    }

    // TODO: Deprecate. Replaced with mesh_tasks
    unsafe fn draw(&mut self, vertices: &[Vertex], triangles: &[GpuTriangle]) -> Result<()> {
        // Not helpful
        // let texture = &self.tex_units[0];
        // gl::UseProgram(texture.program_id);

        // println!("### GL3 draw");
        // Turn the provided vertex data into stored vertex data
        vertices.iter().for_each(|vertex| {
            self.vertices.push(vertex.pos.x);
            self.vertices.push(vertex.pos.y);
            let tex_pos = vertex.tex_pos.unwrap_or(Vector::ZERO);
            self.vertices.push(tex_pos.x);
            self.vertices.push(tex_pos.y);
            self.vertices.push(vertex.col.r);
            self.vertices.push(vertex.col.g);
            self.vertices.push(vertex.col.b);
            self.vertices.push(vertex.col.a);
            self.vertices.push(if vertex.tex_pos.is_some() { 1f32 } else { 0f32 });
        });
        let vertex_length = size_of::<f32>() * self.vertices.len();
        // If the GPU can't store all of our data, re-create the GPU buffers so they can
        if vertex_length > self.vertex_length {
            log::debug!("### vertex_length new={:?} was={:?}", vertex_length, self.vertex_length);
            self.vertex_length = vertex_length * 2;
            // Create strings for all of the shader attributes
            let position_string = CString::new("position").expect("No interior null bytes in shader").into_raw();
            let tex_coord_string = CString::new("tex_coord").expect("No interior null bytes in shader").into_raw();
            let color_string = CString::new("color").expect("No interior null bytes in shader").into_raw();
            let tex_string = CString::new("tex").expect("No interior null bytes in shader").into_raw();
            let use_texture_string = CString::new("uses_texture").expect("No interior null bytes in shader").into_raw();
            // Create the vertex array
            gl::BufferData(gl::ARRAY_BUFFER, self.vertex_length as isize, nullptr(), gl::STREAM_DRAW);

            let stride_distance = (VERTEX_SIZE * size_of::<f32>()) as i32;
            // Set up the vertex attributes
            let pos_attrib = gl::GetAttribLocation(self.shader, position_string as *const i8) as u32;
            gl::EnableVertexAttribArray(pos_attrib);
            gl::VertexAttribPointer(pos_attrib, 2, gl::FLOAT, gl::FALSE, stride_distance, nullptr());

            let tex_attrib = gl::GetAttribLocation(self.shader, tex_coord_string as *const i8) as u32;
            gl::EnableVertexAttribArray(tex_attrib);
            gl::VertexAttribPointer(tex_attrib, 2, gl::FLOAT, gl::FALSE, stride_distance, (2 * size_of::<f32>()) as *const c_void);

            let col_attrib = gl::GetAttribLocation(self.shader, color_string as *const i8) as u32;
            gl::EnableVertexAttribArray(col_attrib);
            gl::VertexAttribPointer(col_attrib, 4, gl::FLOAT, gl::FALSE, stride_distance, (4 * size_of::<f32>()) as *const c_void);

            let use_texture_attrib = gl::GetAttribLocation(self.shader, use_texture_string as *const i8) as u32;
            gl::EnableVertexAttribArray(use_texture_attrib);
            gl::VertexAttribPointer(use_texture_attrib, 1, gl::FLOAT, gl::FALSE, stride_distance, (8 * size_of::<f32>()) as *const c_void);

            // Make sure to deallocate the attribute strings
            CString::from_raw(position_string);
            CString::from_raw(tex_coord_string);
            CString::from_raw(color_string);
            CString::from_raw(tex_string);
            CString::from_raw(use_texture_string);
            //
        }
        let vertex_data = self.vertices.as_ptr() as *const c_void;
        gl::BufferSubData(gl::ARRAY_BUFFER, 0, vertex_length as isize, vertex_data);
        //

        // Scan through the triangles, adding the indices to the index buffer (every time the
        // texture switches, flush and switch the bound texture)
        for triangle in triangles.iter() {
            if let Some(ref img) = triangle.image {
                if self.texture != NULL_TEXTURE_ID && self.texture != img.get_id() {
                    self.flush();
                }
                self.texture = img.get_id();

            }
            self.indices.extend(triangle.indices.iter());
        }
        // Flush any remaining triangles
        self.flush()?;
        self.vertices.clear();
        Ok(())
    }

    // TODO: Deprecate. Replaced with mesh_tasks
    unsafe fn flush(&mut self) -> Result<()> {
        // println!("### GL3 flush");
        if self.indices.len() != 0 {
            // Check if the index buffer is big enough and upload the data
            let index_length = size_of::<u32>() * self.indices.len();
            let index_data = self.indices.as_ptr() as *const c_void;
            if index_length > self.index_length {
                log::debug!("### index_length new={:?} was={:?}", index_length, self.index_length);
                self.index_length = index_length * 2;
                gl::BufferData(gl::ELEMENT_ARRAY_BUFFER, self.index_length as isize, nullptr(), gl::STREAM_DRAW);
            }
            gl::BufferSubData(gl::ELEMENT_ARRAY_BUFFER, 0, index_length as isize, index_data);
            // Upload the texture to the GPU
            gl::ActiveTexture(gl::TEXTURE0);
            if self.texture != 0 {
                // log::debug!("### flush, BindTexture={:?} location={:?}", self.texture, self.texture_location);
                gl::BindTexture(gl::TEXTURE_2D, self.texture);
                gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, self.texture_mode as i32);
                gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, self.texture_mode as i32);
            }
            gl::Uniform1i(self.texture_location, 0);
            // Draw the triangles
            gl::DrawElements(gl::TRIANGLES, self.indices.len() as i32, gl::UNSIGNED_INT, nullptr());

            self.indices.clear();
            self.texture = NULL_TEXTURE_ID;

        }
        Ok(())
    }

    unsafe fn create_texture(&mut self, data: &[u8], width: u32, height: u32, format: PixelFormat) -> Result<ImageData> {
        let data = if data.len() == 0 { nullptr() } else { data.as_ptr() as *const c_void };
        let format = format_gl(format);
        let id = {
            let mut texture = 0;
            gl::GenTextures(1, &mut texture as *mut u32);
            texture
        };
        log::debug!("### Created texture id={} width={:?} height={:?}", id, width, height);
        // gl::ActiveTexture(gl::TEXTURE0 as u32);
        gl::BindTexture(gl::TEXTURE_2D, id);  // WARN: Enabling this makes mesh_tasks fail
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
        gl::TexImage2D(gl::TEXTURE_2D, 0, format as i32, width as i32,
                        height as i32, 0, format, gl::UNSIGNED_BYTE, data);
        // gl::Enable(gl::TEXTURE_2D);  // triggers OpenGL warning
        // Note: this call is not necessary, but help some use cases.
        gl::GenerateMipmap(gl::TEXTURE_2D);
        Ok(ImageData { id, width, height })
    }

    unsafe fn destroy_texture(&mut self, data: &mut ImageData) {
        log::debug!("Destroying texture: {:?}", data.id);
        gl::DeleteTextures(1, &data.id as *const u32);
    }

    unsafe fn create_surface(&mut self, image: &Image) -> Result<SurfaceData> {
        let surface = SurfaceData {
            framebuffer: {
                let mut buffer = 0;
                gl::GenFramebuffers(1, &mut buffer as *mut u32);
                buffer
            }
        };
        gl::BindFramebuffer(gl::FRAMEBUFFER, surface.framebuffer);
        gl::FramebufferTexture(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, image.get_id(), 0);
        gl::DrawBuffers(1, &gl::COLOR_ATTACHMENT0 as *const u32);
        Ok(surface)
    }

    unsafe fn bind_surface(&mut self, surface: &Surface) {
        gl::BindFramebuffer(gl::FRAMEBUFFER, surface.data.framebuffer);
        gl::Viewport(0, 0, surface.image.source_width() as i32, surface.image.source_height() as i32);
    }

    unsafe fn unbind_surface(&mut self, _surface: &Surface, viewport: &[i32]) {
        gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
        gl::Viewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    }

    unsafe fn destroy_surface(&mut self, surface: &SurfaceData) {
        gl::DeleteFramebuffers(1, &surface.framebuffer as *const u32);
    }

    unsafe fn viewport(&self) -> [i32; 4] {
        let mut viewport = [0, 0, 0, 0];
        gl::GetIntegerv(gl::VIEWPORT, (&mut viewport).as_mut_ptr());
        viewport
    }

    unsafe fn set_viewport(&mut self, area: Rectangle) where Self: Sized {
        let size: LogicalSize = area.size().into();
        let dpi = self.context.get_hidpi_factor();
        self.context.resize(size.to_physical(dpi));
        let dpi = dpi as f32;
        gl::Viewport(
            (area.x() * dpi) as i32,
            (area.y() * dpi) as i32,
            (area.width() * dpi) as i32,
            (area.height() * dpi) as i32
        );
    }

    unsafe fn screenshot(&self, format: PixelFormat) -> (Vector, Vec<u8>) where Self: Sized {
        let bytes_per_pixel = match format {
            PixelFormat::RGBA => 4,
            PixelFormat::RGB => 3,
            PixelFormat::Alpha => 1,
        };
        let format = format_gl(format);
        let [x, y, width, height] = self.viewport();
        let length = (width * height * bytes_per_pixel) as usize;
        let mut buffer = Vec::with_capacity(length);
        let pointer = buffer.as_mut_ptr() as *mut c_void;
        gl::ReadPixels(x, y, width, height, format, gl::UNSIGNED_BYTE, pointer);
        buffer.set_len(length);
        (Vector::new(width, height), buffer)
    }

    unsafe fn capture(&self, rect: &Rectangle, format: PixelFormat) -> (Vector, Vec<u8>) where Self: Sized {
        let bytes_per_pixel = byte_size(format);
        let format = format_gl(format);
        let length = (rect.width() * rect.height() * bytes_per_pixel as f32) as usize;
        let mut buffer = Vec::with_capacity(length);
        let pointer = buffer.as_mut_ptr() as *mut c_void;
        let (x, y, width, height) = (rect.x() as i32, rect.y() as i32, rect.width() as i32, rect.height() as i32);
        gl::ReadPixels(x, y, width, height, format, gl::UNSIGNED_BYTE, pointer);
        buffer.set_len(length);
        (Vector::new(width, height), buffer)
    }

    fn set_cursor(&mut self, cursor: MouseCursor) {
        match cursor.into_gl_cursor() {
            Some(gl_cursor) => {
                self.context.hide_cursor(false);
                self.context.set_cursor(gl_cursor);
            }
            None => self.context.hide_cursor(true),
        }
    }

    fn set_title(&mut self, title: &str) {
        self.context.set_title(title);
    }

    fn present(&self) -> Result<()> {
        Ok(self.context.swap_buffers()?)
    }

    fn set_fullscreen(&mut self, fullscreen: bool) -> Option<Vector> {
        self.context.set_fullscreen(if fullscreen {
            Some(self.context.get_primary_monitor())
        } else {
            None
        });
        None
    }

    fn resize(&mut self, size: Vector) {
        self.context.set_inner_size(size.into());
    }

    /// Create and register a TextureUnit in self.tex_units given the Texture object which
    /// contains all of the parameters needed. This does not create or upload a texture, which
    /// is a secondary step.
    fn create_texture_unit(&mut self, texture: &Texture) -> Result<(usize)> {
        let pointer = self.prepare_texture(&texture.vertex_shader, &texture.fragment_shader)?;
        self.configure_texture(pointer, &texture.fields, serialize_vertex, &texture.out_color, &texture.sampler)?;

        Ok(pointer)
    }

    fn prepare_texture(&mut self, vertex_shader: &str, fragment_shader: &str) -> Result<usize> {

        unsafe {
            // gl::BindTexture(gl::TEXTURE_2D, 0);
            gl::BindBuffer(gl::ARRAY_BUFFER, self.vbo);
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, self.ebo);

            let vertex_id = self.compile_shader(vertex_shader, gl::VERTEX_SHADER).unwrap();
            let fragment_id = self.compile_shader(fragment_shader, gl::FRAGMENT_SHADER).unwrap();
            let program_id = self.link_program(vertex_id, fragment_id).unwrap();
            let texture_id = 0;  // Initially set to 0. Will be assigned in upload_texture

            let idx = self.tex_units.len();
            log::debug!("==Prepare=========================================================");
            // log::debug!(">>> Globals vao={} vbo={} ebo={}", self.vao, self.vbo, self.ebo);
            log::debug!(">>> Created program_id={}", program_id);
            // log::debug!(">>> vertex_id={} fragment_id={}", vertex_id, fragment_id);

            // Create a no-op serializer function
            let serializer = |_vertex| -> Vec<f32> {
                Vec::new()
            };

            let unit = TextureUnit {
                program_id,
                vertex_id,
                fragment_id,
                texture_id,
                location_id: 0,
                serializer: Box::new(serializer),
            };

            self.tex_units.push(unit);
            return Ok(idx);
        }
    }

    fn configure_texture<CB>(&mut self, idx: usize, fields: &Vec<(String, u32)>, cb: CB, out_color: &str, tex_name: &str) -> Result<()>
    where CB: Fn(Vertex) -> Vec<f32> + 'static,
    {
        if idx >= self.tex_units.len() {
            let message = format!("Texture index {} out of bounds for len={}", idx, self.tex_units.len());
            return Err(QuicksilverError::ContextError(message));
        }
        let texture = &mut self.tex_units[idx];
        let program_id = texture.program_id;
        self.tex_units[idx].serializer = Box::new(cb);

        let float_size = size_of::<f32>() as u32;
        let vert_size = fields.iter().fold(0, |acc, x| acc + x.1);
        let stride_distance = (vert_size * float_size) as i32;
        log::debug!("Configuring texture idx={}, program_id={} vert_size={} float_size={}", idx, program_id, vert_size, float_size);

        unsafe {

            let raw = CString::new(tex_name).expect("No color name").into_raw();
            let location = gl::GetUniformLocation(program_id, raw as *mut i8);
            self.tex_units[idx].location_id = location;
            log::debug!(">>> texture location={:?} for program_id={:?}", location, program_id);
            CString::from_raw(raw);

            // Map the out_color variable name to the fragment shader output
            let raw = CString::new(out_color).expect("No color name").into_raw();

            gl::BindFragDataLocation(program_id, idx as u32, raw as *mut i8);
            CString::from_raw(raw);

            let mut offset = 0;
            for (v_field, float_count) in fields {
                let count = *float_count;
                log::debug!("[{:?}] stride_distance={:?} offset={:?}", &v_field, stride_distance, offset);
                let c_name = CString::new(v_field.to_string()).expect("No interior null bytes in shader").into_raw();
                let attr = gl::GetAttribLocation(program_id, c_name as *const i8);
                CString::from_raw(c_name);
                if attr < 0 {
                    log::debug!("configure_fields error: attr={:?} y={:?}", attr, 0);
                    return Err(QuicksilverError::ContextError(format!("{} GetAttribLocation -> {}", v_field, attr)));
                }

                gl::EnableVertexAttribArray(attr as u32);
                gl::VertexAttribPointer(
                    attr as u32,
                    count as i32,
                    gl::FLOAT,
                    gl::FALSE,
                    stride_distance,
                    offset as _,
                );

                offset += count * float_size;
            }
            gl::UseProgram(0);

            Ok(())
        }
    }

    // TODO: Consolidate with create_texture method
    fn upload_texture(&mut self, idx: usize, data: &[u8], width: u32, height: u32, format: PixelFormat) -> Result<()> {
        unsafe {
            if idx >= self.tex_units.len() {
                let message = format!("Texture index {} out of bounds for len={}", idx, self.tex_units.len());
                return Err(QuicksilverError::ContextError(message));
            }

            let texture = &mut self.tex_units[idx];
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);

            let gl_format = format_gl(format);
            // let gl_bytes = byte_size(format);

            gl::BindTexture(gl::TEXTURE_2D, 0);

            // This 1 value only valid for single channel (RED). https://www.khronos.org/opengl/wiki/Common_Mistakes
            // gl::PixelStorei(gl::UNPACK_ALIGNMENT, gl_bytes as i32);

            let mut texture_id = 0;
            // https://www.khronos.org/opengl/wiki/GLSL_Sampler#Binding_textures_to_samplers
            gl::ActiveTexture(gl::TEXTURE0 + idx as u32);
            gl::GenTextures(1, &mut texture_id);
            gl::BindTexture(gl::TEXTURE_2D, texture_id);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
            // gl::Enable(gl::TEXTURE_2D);
            if texture_id > 0 {
                log::debug!("{}/ >>> Created texture_id={}", idx, texture_id);
            } else {
                log::debug!("==ERROR in GenTextures=========================================================");
            }

            // Save the new id for later use
            texture.texture_id = texture_id;
            let data = if data.len() == 0 { nullptr() } else { data.as_ptr() as *const c_void };
            gl::TexImage2D(gl::TEXTURE_2D, 0, gl_format as i32, width as i32,
                            height as i32, 0, gl_format, gl::UNSIGNED_BYTE, data);

            // Note: this call is not necessary, but help some use cases.
            // gl::GenerateMipmap(gl::TEXTURE_2D);

            let mut tex_width: i32 = 0;
            let mut tex_height: i32 = 0;
            gl::GetTexLevelParameteriv(gl::TEXTURE_2D, 0, gl::TEXTURE_WIDTH, &mut tex_width);
            gl::GetTexLevelParameteriv(gl::TEXTURE_2D, 0, gl::TEXTURE_HEIGHT, &mut tex_height);

            log::debug!(">>> TEX width={:?} height={:?}", tex_width, tex_height);
            gl::UseProgram(0);

            Ok(())
        }
    }

    fn update_texture(&mut self, idx: usize, data: &[u8], rect: &Rectangle, format: PixelFormat) -> Result<()> {
        if idx >= self.tex_units.len() {
            let message = format!("Texture index {} out of bounds for len={}", idx, self.tex_units.len());
            return Err(QuicksilverError::ContextError(message));
        }
        let texture = &self.tex_units[idx];
        let tex_id = texture.texture_id;

        let gl_format = format_gl(format);
        // let gl_bytes = byte_size(format);
        // log::debug!("Updating [{}] texture_id={:?} rect={:?} format={:?}", idx, id, rect, gl_format);

        unsafe {

            gl::UseProgram(texture.program_id);

            gl::ActiveTexture(gl::TEXTURE0 + idx as u32);
            // https://www.khronos.org/opengl/wiki/GLAPI/glTexSubImage2D
            gl::BindTexture(gl::TEXTURE_2D, tex_id);
            gl::Uniform1i(texture.location_id, idx as i32);

            gl::TexSubImage2D(
                gl::TEXTURE_2D,
                0,
                rect.x() as _,
                rect.y() as _,
                rect.width() as _,
                rect.height() as _,
                gl_format,
                gl::UNSIGNED_BYTE,
                data.as_ptr() as _,
            );
            // gl::UseProgram(0);

            Ok(())
        }
    }

    /// The logic in this method handles the overly complex situation where all of the vertices and triangles
    /// that were accumulated in Mesh are batched together.
    unsafe fn execute_tasks(&mut self, tasks: &Vec<MeshTask>) -> Result<()> {
        for (_, task) in tasks.iter().enumerate() {

            if task.pointer >= self.tex_units.len() {
                log::debug!("Texture index {} out of bounds for len={}", task.pointer, self.tex_units.len());
                continue;
            }
            let idx = task.pointer as u32;
            let texture = &self.tex_units[task.pointer];
            // let texture_id = texture.texture_id;
            gl::ActiveTexture(gl::TEXTURE0 + idx);
            gl::UseProgram(texture.program_id);
            gl::Uniform1i(texture.location_id, idx as i32);

            let mut vertices: Vec<f32> = Vec::new();
            let mut cb = &texture.serializer;
            for vertex in &task.vertices {
                let mut verts = (&mut cb)(*vertex);
                vertices.append(&mut verts);
            }
            let vertex_length = size_of::<f32>() * vertices.len();
            if vertex_length > self.vertex_length {
                log::debug!("{}/ >>> vertex_length new={:?} was={:?}", idx, vertex_length, self.vertex_length);
                self.vertex_length = vertex_length * 2;
                // Create the vertex array
                gl::BufferData(gl::ARRAY_BUFFER, self.vertex_length as isize, nullptr(), gl::STATIC_DRAW);
            }
            let vertex_data = vertices.as_ptr() as *const c_void;
            gl::BufferSubData(gl::ARRAY_BUFFER, 0, vertex_length as isize, vertex_data);

            let ranges: Vec<(Option<u32>, Range<usize>)> = {

                let mut ranges: Vec<(Option<u32>, Range<usize>)> = Vec::new();
                if task.pointer == 0 {
                    let mut last_id: Option<u32> = None;
                    let mut range_start: usize = 0;

                    for (i, triangle) in task.triangles.iter().enumerate() {
                        let img_id: Option<u32> = {
                            if let Some(ref img) = triangle.image {
                                Some(img.get_id())
                            } else {
                                None
                            }
                        };
                        if img_id != last_id {
                            // log::debug!("img_id changed new={:?} was={:?}", img_id, last_id);
                            let range: Range<usize> = range_start..i;
                            ranges.push((last_id, range));
                            range_start = i;
                            last_id = img_id;
                        }
                    }
                    let range: Range<usize> = range_start..task.triangles.len();
                    ranges.push((last_id, range));
                    ranges
                } else {
                    let range: Range<usize> = 0..task.triangles.len();
                    ranges.push((Some(texture.texture_id), range));
                    ranges
                }
            };

            for data in &ranges {
                let range = data.1.clone();
                let mut indices: Vec<u32> = Vec::new();
                for triangle in &task.triangles[range] {
                    // log::debug!("add indices={:?} range={:?}", &triangle.indices, data.1.clone());
                    indices.extend_from_slice(&triangle.indices);
                }
                let index_length = size_of::<u32>() * indices.len();
                let index_data = indices.as_ptr() as *const c_void;
                // If the GPU can't store all of our data, re-create the GPU buffers so they can
                if index_length > self.index_length {
                    log::debug!("{}/ >>> index_length new={:?} was={:?}", idx, index_length, self.index_length);
                    self.index_length = index_length * 2;
                    gl::BufferData(gl::ELEMENT_ARRAY_BUFFER, self.index_length as isize, nullptr(), gl::STATIC_DRAW);
                }
                gl::BufferSubData(gl::ELEMENT_ARRAY_BUFFER, 0, index_length as isize, index_data);

                let texture_id: u32 = {
                    if let Some(id) = data.0 {
                        id
                    } else {
                        texture.texture_id
                    }
                };

                if gl::IsTexture(texture_id) == gl::TRUE {
                    gl::BindTexture(gl::TEXTURE_2D, texture_id);
                } else {
                    // log::debug!("execute_tasks {:?} is NOT a texture", texture_id);
                }

                gl::Uniform1i(texture.location_id, idx as i32);
                gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, self.texture_mode as i32);
                gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, self.texture_mode as i32);

                // Draw the triangles
                gl::DrawElements(gl::TRIANGLES, indices.len() as i32, gl::UNSIGNED_INT, nullptr());
            }
        }
        Ok(())
    }

    fn reset_gpu(&mut self) {
        unsafe {
            for (i, texture) in self.tex_units.iter().enumerate() {
                if i > 0 {
                    gl::DeleteTextures(1, texture.texture_id as *const u32);
                    gl::DeleteProgram(texture.program_id);
                    gl::DeleteShader(texture.fragment_id);
                    gl::DeleteShader(texture.vertex_id);
                }
            }
        }
    }
}

impl GL3Backend {

    /// Returns the u32 id value of the compiled shader
    fn compile_shader(&self, src: &str, stype: GLenum) -> Result<u32> {
        let shader;
        unsafe {
            shader = gl::CreateShader(stype);

            // Attempt to compile the shader
            let c_str = CString::new(src.as_bytes()).expect("No interior null bytes in shader").into_raw();
            gl::ShaderSource(shader, 1, &(c_str as *const i8) as *const *const i8, nullptr());
            CString::from_raw(c_str);
            gl::CompileShader(shader);


            // Get the compile status
            let mut status = GLint::from(gl::FALSE);
            gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut status);

            // Fail on error
            if status != GLint::from(gl::TRUE) {
                let mut len = 0;
                gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
                let mut buf = Vec::with_capacity(len as usize);
                buf.set_len((len as usize) - 1); // subtract 1 to skip the trailing null character
                gl::GetShaderInfoLog(
                    shader,
                    len,
                    ptr::null_mut(),
                    buf.as_mut_ptr() as *mut GLchar,
                );
                return Err(QuicksilverError::ContextError(String::from_utf8(buf).unwrap()));
            }

        }
        Ok(shader)
    }

    unsafe fn link_program(&self, vs: u32, fs: u32) -> Result<u32> {
        let program = gl::CreateProgram();
        gl::AttachShader(program, vs);
        gl::AttachShader(program, fs);
        gl::LinkProgram(program);
        gl::UseProgram(program);

        let raw = CString::new("font_tex").expect("No color name").into_raw();
        let location = gl::GetUniformLocation(program, raw as *mut i8);
        log::debug!(">>> link_program texture location={:?} for program_id={:?}", location, program);
        CString::from_raw(raw);

        // Get the link status
        let mut status = GLint::from(gl::FALSE);
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut status);

        // Fail on error
        if status != GLint::from(gl::TRUE) {
            let mut len: GLint = 0;
            gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
            let mut buf = Vec::with_capacity(len as usize);
            buf.set_len((len as usize) - 1); // subtract 1 to skip the trailing null character
            gl::GetProgramInfoLog(
                program,
                len,
                ptr::null_mut(),
                buf.as_mut_ptr() as *mut GLchar,
            );
            return Err(QuicksilverError::ContextError(String::from_utf8(buf).unwrap()));
        }

        Ok(program)
    }
    // unsafe fn default_texture_unit

}

impl Drop for GL3Backend {
    fn drop(&mut self) {
        unsafe {
            self.reset_gpu();
            gl::DeleteProgram(self.shader);
            gl::DeleteShader(self.fragment);
            gl::DeleteShader(self.vertex);
            gl::DeleteBuffers(2, &[self.vbo, self.ebo] as *const u32);
            gl::DeleteVertexArrays(1, &self.vao as *const u32);
        }
    }
}

// *****************************************************************************************************
// Default shader constants
// *****************************************************************************************************

const DEFAULT_VERTEX_SHADER: &str = r#"#version 150
in vec2 position;
in vec2 tex_coord;
in vec4 color;
in float uses_texture;
out vec4 Color;
out vec2 Tex_coord;
out float Uses_texture;
void main() {
    Color = color;
    Tex_coord = tex_coord;
    Uses_texture = uses_texture;
    gl_Position = vec4(position, 0, 1);
}"#;

const DEFAULT_FRAGMENT_SHADER: &str = r#"#version 150
in vec4 Color;
in vec2 Tex_coord;
in float Uses_texture;
out vec4 outColor;
uniform sampler2D tex;
void main() {
    vec4 tex_color = (Uses_texture != 0) ? texture(tex, Tex_coord) : vec4(1, 1, 1, 1);
    outColor = Color * tex_color;
}"#;

const TEX_FIELDS: &[(&str, u32)] = &[
            ("position", 2),
            ("tex_coord", 2),
            ("color", 4),
            ("uses_texture", 1),
        ];
const OUT_COLOR: &str = "outColor";
const SAMPLER: &str = "tex";

fn serialize_vertex(vertex: Vertex) -> Vec<f32> {
    let mut result: Vec<f32> = Vec::new();
    result.push(vertex.pos.x);
    result.push(vertex.pos.y);
    let tex_pos = vertex.tex_pos.unwrap_or(Vector::ZERO);
    result.push(tex_pos.x);
    result.push(tex_pos.y);
    result.push(vertex.col.r);
    result.push(vertex.col.g);
    result.push(vertex.col.b);
    result.push(vertex.col.a);
    result.push(if vertex.tex_pos.is_some() { 1f32 } else { 0f32 });
    result
}


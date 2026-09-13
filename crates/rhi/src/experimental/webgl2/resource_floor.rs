//! Closed WebGL2 common-resource conformance witness types.
//!
//! The session owns the corresponding browser-object lease; this module keeps
//! its public evidence separate from the graph-executor capability profile.

use super::*;

/// Deterministic observations from the closed WebGL2 common-resource recipe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WebGl2ResourceFloorEvidence {
    /// RGBA8 pixel sampled to the default framebuffer by the second pass.
    pub sampled_default_pixel: [u8; 4],
    /// Bytes uploaded, copied with `copyBufferSubData`, and read back.
    pub copied_buffer_bytes: [u8; 4],
    /// RGBA8 pixel copied with `copyTexSubImage2D` and read from its FBO.
    pub copied_texture_pixel: [u8; 4],
    /// The fixed `DEPTH_COMPONENT32F` attachment made its framebuffer complete.
    pub depth32float_attachment_usable: bool,
}

pub(super) struct ResourceFloorObjects {
    pub(super) offscreen_program: WebGlProgram,
    pub(super) present_program: WebGlProgram,
    pub(super) vertex: WebGlBuffer,
    pub(super) index: WebGlBuffer,
    pub(super) uniform: WebGlBuffer,
    pub(super) vertex_array: WebGlVertexArrayObject,
    pub(super) source_sampler_uniform: WebGlUniformLocation,
    pub(super) sampler_uniform: WebGlUniformLocation,
    pub(super) source_buffer: WebGlBuffer,
    pub(super) copied_buffer: WebGlBuffer,
    pub(super) uploaded_color: WebGlTexture,
    pub(super) offscreen_color: WebGlTexture,
    pub(super) copied_color: WebGlTexture,
    pub(super) depth: WebGlTexture,
    pub(super) offscreen_framebuffer: WebGlFramebuffer,
    pub(super) copied_framebuffer: WebGlFramebuffer,
    pub(super) sampler: WebGlSampler,
    pub(super) evidence: WebGl2ResourceFloorEvidence,
}

pub(super) fn create(gl: &Gl) -> Result<ResourceFloorObjects, String> {
    let offscreen_program = create_program(
        gl,
        "#version 300 es\nlayout(location=0) in vec2 a_position; out vec2 v_uv; void main(){ gl_Position=vec4(a_position,0.0,1.0); v_uv=a_position*0.5+0.5; }",
        "#version 300 es\nprecision mediump float; in vec2 v_uv; uniform sampler2D u_uploaded; layout(std140) uniform FixtureTint { vec4 u_color; }; out vec4 out_color; void main(){ out_color=texture(u_uploaded,v_uv)*u_color; }",
    )?;
    let present_program = match create_program(
        gl,
        "#version 300 es\nlayout(location=0) in vec2 a_position; out vec2 v_uv; void main(){ gl_Position=vec4(a_position,0.0,1.0); v_uv=a_position*0.5+0.5; }",
        "#version 300 es\nprecision mediump float; in vec2 v_uv; uniform sampler2D u_color; out vec4 out_color; void main(){ out_color=texture(u_color,v_uv); }",
    ) {
        Ok(value) => value,
        Err(error) => {
            gl.delete_program(Some(&offscreen_program));
            return Err(error);
        }
    };
    let uniform_block = gl.get_uniform_block_index(&offscreen_program, "FixtureTint");
    if uniform_block == Gl::INVALID_INDEX {
        gl.delete_program(Some(&offscreen_program));
        gl.delete_program(Some(&present_program));
        return Err("resource-floor FixtureTint block missing".into());
    }
    gl.uniform_block_binding(&offscreen_program, uniform_block, 0);
    let source_sampler_uniform = match gl.get_uniform_location(&offscreen_program, "u_uploaded") {
        Some(value) => value,
        None => {
            gl.delete_program(Some(&offscreen_program));
            gl.delete_program(Some(&present_program));
            return Err("resource-floor source sampler uniform missing".into());
        }
    };
    let sampler_uniform = match gl.get_uniform_location(&present_program, "u_color") {
        Some(value) => value,
        None => {
            gl.delete_program(Some(&offscreen_program));
            gl.delete_program(Some(&present_program));
            return Err("resource-floor sampler uniform missing".into());
        }
    };
    let allocated = (
        gl.create_buffer(),
        gl.create_buffer(),
        gl.create_buffer(),
        gl.create_buffer(),
        gl.create_buffer(),
        gl.create_vertex_array(),
        gl.create_texture(),
        gl.create_texture(),
        gl.create_texture(),
        gl.create_texture(),
        gl.create_framebuffer(),
        gl.create_framebuffer(),
        gl.create_sampler(),
    );
    let (
        vertex,
        index,
        source_buffer,
        copied_buffer,
        uniform,
        vertex_array,
        uploaded_color,
        offscreen_color,
        copied_color,
        depth,
        offscreen_framebuffer,
        copied_framebuffer,
        sampler,
    ) = match allocated {
        (
            Some(vertex),
            Some(index),
            Some(source_buffer),
            Some(copied_buffer),
            Some(uniform),
            Some(vertex_array),
            Some(uploaded_color),
            Some(offscreen_color),
            Some(copied_color),
            Some(depth),
            Some(offscreen_framebuffer),
            Some(copied_framebuffer),
            Some(sampler),
        ) => (
            vertex,
            index,
            source_buffer,
            copied_buffer,
            uniform,
            vertex_array,
            uploaded_color,
            offscreen_color,
            copied_color,
            depth,
            offscreen_framebuffer,
            copied_framebuffer,
            sampler,
        ),
        (
            vertex,
            index,
            source_buffer,
            copied_buffer,
            uniform,
            vertex_array,
            uploaded_color,
            offscreen_color,
            copied_color,
            depth,
            offscreen_framebuffer,
            copied_framebuffer,
            sampler,
        ) => {
            for buffer in [
                vertex.as_ref(),
                index.as_ref(),
                source_buffer.as_ref(),
                copied_buffer.as_ref(),
                uniform.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                gl.delete_buffer(Some(buffer));
            }
            if let Some(vertex_array) = vertex_array.as_ref() {
                gl.delete_vertex_array(Some(vertex_array));
            }
            for texture in [
                offscreen_color.as_ref(),
                uploaded_color.as_ref(),
                copied_color.as_ref(),
                depth.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                gl.delete_texture(Some(texture));
            }
            for framebuffer in [offscreen_framebuffer.as_ref(), copied_framebuffer.as_ref()]
                .into_iter()
                .flatten()
            {
                gl.delete_framebuffer(Some(framebuffer));
            }
            if let Some(sampler) = sampler.as_ref() {
                gl.delete_sampler(Some(sampler));
            }
            gl.delete_program(Some(&offscreen_program));
            gl.delete_program(Some(&present_program));
            return Err("create resource-floor object returned null".into());
        }
    };
    let mut objects = ResourceFloorObjects {
        offscreen_program,
        present_program,
        vertex,
        index,
        uniform,
        vertex_array,
        source_sampler_uniform,
        sampler_uniform,
        source_buffer,
        copied_buffer,
        uploaded_color,
        offscreen_color,
        copied_color,
        depth,
        offscreen_framebuffer,
        copied_framebuffer,
        sampler,
        evidence: WebGl2ResourceFloorEvidence {
            sampled_default_pixel: [0; 4],
            copied_buffer_bytes: [0; 4],
            copied_texture_pixel: [0; 4],
            depth32float_attachment_usable: false,
        },
    };
    let result = run(gl, &mut objects);
    if let Err(error) = result {
        destroy(gl, objects);
        return Err(error);
    }
    Ok(objects)
}

pub(super) fn run(gl: &Gl, objects: &mut ResourceFloorObjects) -> Result<(), String> {
    const WITNESS: [u8; 4] = [51, 102, 153, 255];
    // The witness is byte-exact. Disable state that could intentionally alter
    // a representable RGBA8 result, and exercise the depth attachment in draw.
    gl.disable(Gl::DITHER);
    gl.disable(Gl::BLEND);
    gl.enable(Gl::DEPTH_TEST);
    gl.bind_buffer(Gl::COPY_READ_BUFFER, Some(&objects.source_buffer));
    gl.buffer_data_with_u8_array(Gl::COPY_READ_BUFFER, &WITNESS, Gl::STATIC_DRAW);
    gl.bind_buffer(Gl::COPY_WRITE_BUFFER, Some(&objects.copied_buffer));
    gl.buffer_data_with_i32(Gl::COPY_WRITE_BUFFER, 4, Gl::STATIC_DRAW);
    gl.copy_buffer_sub_data_with_i32_and_i32_and_i32(
        Gl::COPY_READ_BUFFER,
        Gl::COPY_WRITE_BUFFER,
        0,
        0,
        4,
    );
    let mut copied_buffer_bytes = [0; 4];
    gl.get_buffer_sub_data_with_i32_and_u8_array(
        Gl::COPY_WRITE_BUFFER,
        0,
        &mut copied_buffer_bytes,
    );
    if copied_buffer_bytes != WITNESS {
        return Err("buffer copy readback did not preserve the witness".into());
    }

    let vertices = Float32Array::from(&[-1.0_f32, -1.0, 3.0, -1.0, -1.0, 3.0][..]);
    gl.bind_vertex_array(Some(&objects.vertex_array));
    gl.bind_buffer(Gl::ARRAY_BUFFER, Some(&objects.vertex));
    gl.buffer_data_with_array_buffer_view(Gl::ARRAY_BUFFER, &vertices, Gl::STATIC_DRAW);
    let indices = Uint32Array::from(&[0_u32, 1, 2][..]);
    gl.bind_buffer(Gl::ELEMENT_ARRAY_BUFFER, Some(&objects.index));
    gl.buffer_data_with_array_buffer_view(Gl::ELEMENT_ARRAY_BUFFER, &indices, Gl::STATIC_DRAW);
    let tint = Float32Array::from(&[1.0_f32, 1.0, 1.0, 1.0][..]);
    gl.bind_buffer(Gl::UNIFORM_BUFFER, Some(&objects.uniform));
    gl.buffer_data_with_array_buffer_view(Gl::UNIFORM_BUFFER, &tint, Gl::STATIC_DRAW);
    gl.bind_buffer_base(Gl::UNIFORM_BUFFER, 0, Some(&objects.uniform));
    gl.bind_texture(Gl::TEXTURE_2D, Some(&objects.uploaded_color));
    gl.tex_storage_2d(Gl::TEXTURE_2D, 1, Gl::RGBA8, 1, 1);
    gl.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_u8_array(
        Gl::TEXTURE_2D,
        0,
        0,
        0,
        1,
        1,
        Gl::RGBA,
        Gl::UNSIGNED_BYTE,
        Some(&WITNESS),
    )
    .map_err(|error| format!("upload immutable RGBA8 texture: {error:?}"))?;
    for texture in [&objects.offscreen_color, &objects.copied_color] {
        gl.bind_texture(Gl::TEXTURE_2D, Some(texture));
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MIN_FILTER, Gl::NEAREST as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MAG_FILTER, Gl::NEAREST as i32);
        gl.tex_storage_2d(Gl::TEXTURE_2D, 1, Gl::RGBA8, 1, 1);
    }
    gl.bind_texture(Gl::TEXTURE_2D, Some(&objects.depth));
    gl.tex_storage_2d(Gl::TEXTURE_2D, 1, Gl::DEPTH_COMPONENT32F, 1, 1);
    gl.bind_framebuffer(Gl::FRAMEBUFFER, Some(&objects.offscreen_framebuffer));
    gl.framebuffer_texture_2d(
        Gl::FRAMEBUFFER,
        Gl::COLOR_ATTACHMENT0,
        Gl::TEXTURE_2D,
        Some(&objects.offscreen_color),
        0,
    );
    gl.framebuffer_texture_2d(
        Gl::FRAMEBUFFER,
        Gl::DEPTH_ATTACHMENT,
        Gl::TEXTURE_2D,
        Some(&objects.depth),
        0,
    );
    if gl.check_framebuffer_status(Gl::FRAMEBUFFER) != Gl::FRAMEBUFFER_COMPLETE {
        return Err("Depth32Float offscreen framebuffer is incomplete".into());
    }
    objects.evidence.depth32float_attachment_usable = true;
    gl.viewport(0, 0, 1, 1);
    gl.use_program(Some(&objects.offscreen_program));
    gl.bind_buffer(Gl::ARRAY_BUFFER, Some(&objects.vertex));
    gl.bind_buffer(Gl::ELEMENT_ARRAY_BUFFER, Some(&objects.index));
    gl.vertex_attrib_pointer_with_i32(0, 2, Gl::FLOAT, false, 0, 0);
    gl.enable_vertex_attrib_array(0);
    gl.active_texture(Gl::TEXTURE0);
    gl.bind_texture(Gl::TEXTURE_2D, Some(&objects.uploaded_color));
    gl.sampler_parameteri(&objects.sampler, Gl::TEXTURE_MIN_FILTER, Gl::NEAREST as i32);
    gl.sampler_parameteri(&objects.sampler, Gl::TEXTURE_MAG_FILTER, Gl::NEAREST as i32);
    gl.bind_sampler(0, Some(&objects.sampler));
    gl.uniform1i(Some(&objects.source_sampler_uniform), 0);
    gl.clear_color(0.0, 0.0, 0.0, 1.0);
    gl.clear(Gl::COLOR_BUFFER_BIT | Gl::DEPTH_BUFFER_BIT);
    gl.draw_elements_with_i32(Gl::TRIANGLES, 3, Gl::UNSIGNED_INT, 0);
    gl.read_buffer(Gl::COLOR_ATTACHMENT0);
    gl.bind_texture(Gl::TEXTURE_2D, Some(&objects.copied_color));
    gl.copy_tex_sub_image_2d(Gl::TEXTURE_2D, 0, 0, 0, 0, 0, 1, 1);
    gl.bind_framebuffer(Gl::FRAMEBUFFER, Some(&objects.copied_framebuffer));
    gl.framebuffer_texture_2d(
        Gl::FRAMEBUFFER,
        Gl::COLOR_ATTACHMENT0,
        Gl::TEXTURE_2D,
        Some(&objects.copied_color),
        0,
    );
    if gl.check_framebuffer_status(Gl::FRAMEBUFFER) != Gl::FRAMEBUFFER_COMPLETE {
        return Err("copied-color framebuffer is incomplete".into());
    }
    gl.read_buffer(Gl::COLOR_ATTACHMENT0);
    let mut copied_texture_pixel = [0; 4];
    gl.read_pixels_with_opt_u8_array(
        0,
        0,
        1,
        1,
        Gl::RGBA,
        Gl::UNSIGNED_BYTE,
        Some(&mut copied_texture_pixel),
    )
    .map_err(|error| format!("read copied texture: {error:?}"))?;
    if copied_texture_pixel != WITNESS {
        return Err("texture copy readback did not preserve the witness".into());
    }

    gl.bind_framebuffer(Gl::FRAMEBUFFER, None);
    gl.viewport(0, 0, 1, 1);
    gl.use_program(Some(&objects.present_program));
    gl.active_texture(Gl::TEXTURE0);
    gl.bind_texture(Gl::TEXTURE_2D, Some(&objects.copied_color));
    gl.sampler_parameteri(&objects.sampler, Gl::TEXTURE_MIN_FILTER, Gl::NEAREST as i32);
    gl.sampler_parameteri(&objects.sampler, Gl::TEXTURE_MAG_FILTER, Gl::NEAREST as i32);
    gl.bind_sampler(0, Some(&objects.sampler));
    gl.uniform1i(Some(&objects.sampler_uniform), 0);
    gl.draw_elements_with_i32(Gl::TRIANGLES, 3, Gl::UNSIGNED_INT, 0);
    let mut sampled_default_pixel = [0; 4];
    gl.read_pixels_with_opt_u8_array(
        0,
        0,
        1,
        1,
        Gl::RGBA,
        Gl::UNSIGNED_BYTE,
        Some(&mut sampled_default_pixel),
    )
    .map_err(|error| format!("read sampled default framebuffer: {error:?}"))?;
    if sampled_default_pixel != WITNESS {
        return Err("sampled default framebuffer did not preserve the witness".into());
    }
    if gl.get_error() != Gl::NO_ERROR {
        return Err("WebGL error after resource-floor recipe".into());
    }
    gl.disable(Gl::DEPTH_TEST);
    objects.evidence = WebGl2ResourceFloorEvidence {
        sampled_default_pixel,
        copied_buffer_bytes,
        copied_texture_pixel,
        depth32float_attachment_usable: true,
    };
    Ok(())
}

pub(super) fn destroy(gl: &Gl, objects: ResourceFloorObjects) {
    gl.bind_sampler(0, None);
    gl.bind_vertex_array(None);
    gl.bind_framebuffer(Gl::FRAMEBUFFER, None);
    gl.bind_texture(Gl::TEXTURE_2D, None);
    gl.disable(Gl::DEPTH_TEST);
    gl.use_program(None);
    gl.delete_sampler(Some(&objects.sampler));
    gl.delete_framebuffer(Some(&objects.offscreen_framebuffer));
    gl.delete_framebuffer(Some(&objects.copied_framebuffer));
    gl.delete_texture(Some(&objects.offscreen_color));
    gl.delete_texture(Some(&objects.uploaded_color));
    gl.delete_texture(Some(&objects.copied_color));
    gl.delete_texture(Some(&objects.depth));
    gl.delete_buffer(Some(&objects.vertex));
    gl.delete_buffer(Some(&objects.index));
    gl.delete_buffer(Some(&objects.source_buffer));
    gl.delete_buffer(Some(&objects.copied_buffer));
    gl.delete_buffer(Some(&objects.uniform));
    gl.delete_vertex_array(Some(&objects.vertex_array));
    gl.delete_program(Some(&objects.offscreen_program));
    gl.delete_program(Some(&objects.present_program));
}

fn create_program(
    gl: &Gl,
    vertex_source: &str,
    fragment_source: &str,
) -> Result<WebGlProgram, String> {
    let vertex = compile(gl, Gl::VERTEX_SHADER, vertex_source)?;
    let fragment = match compile(gl, Gl::FRAGMENT_SHADER, fragment_source) {
        Ok(value) => value,
        Err(error) => {
            gl.delete_shader(Some(&vertex));
            return Err(error);
        }
    };
    let program = match gl.create_program() {
        Some(program) => program,
        None => {
            gl.delete_shader(Some(&vertex));
            gl.delete_shader(Some(&fragment));
            return Err("createProgram returned null".into());
        }
    };
    gl.attach_shader(&program, &vertex);
    gl.attach_shader(&program, &fragment);
    gl.link_program(&program);
    gl.delete_shader(Some(&vertex));
    gl.delete_shader(Some(&fragment));
    if gl
        .get_program_parameter(&program, Gl::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(program)
    } else {
        let message = gl
            .get_program_info_log(&program)
            .unwrap_or_else(|| "program link failed".into());
        gl.delete_program(Some(&program));
        Err(message)
    }
}

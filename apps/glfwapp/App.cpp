#include "App.hpp"
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

void App::run()
{
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

void App::initWindow()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    m_window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
}

void App::initVulkan()
{
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
    createCommandPool();
    createCommandBuffers();
    createSyncObjects();
}

void App::mainLoop()
{
    while (!glfwWindowShouldClose(m_window)) {
        glfwPollEvents();
        drawFrame();
    }

    m_device->waitIdle();
}

void App::cleanup()
{
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        m_device->destroyFence(m_inFlightFences[i]);
    }

    glfwDestroyWindow(m_window);
    glfwTerminate();
}

void App::createInstance()
{
    // get the instance independent function pointers
    static vk::DynamicLoader dl;
    auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    vk::ApplicationInfo appInfo("Hello Triangle", VK_MAKE_VERSION(1, 0, 0),
        "No Engine", VK_MAKE_VERSION(1, 0, 0),
        VK_API_VERSION_1_2);

    auto extensions = getRequiredExtensions();

    if (enableValidationLayers) {
        // in debug mode, use the debugUtilsMessengerCallback
        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);

        vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);

        vk::StructureChain<vk::InstanceCreateInfo,
            vk::DebugUtilsMessengerCreateInfoEXT>
            createInfo({ {}, &appInfo, validationLayers, extensions },
                { {},
                    severityFlags,
                    messageTypeFlags,
                    &debugUtilsMessengerCallback });
        m_instance = vk::createInstanceUnique(createInfo.get<vk::InstanceCreateInfo>());
    } else {
        // in non-debug mode
        vk::InstanceCreateInfo createInfo({}, &appInfo, {}, extensions);
        m_instance = vk::createInstanceUnique(createInfo, nullptr);
    }

    // get all the other function pointers
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*m_instance);
}

void App::setupDebugMessenger()
{
    if (!enableValidationLayers)
        return;

    vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
    vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);

    m_debugUtilsMessenger = m_instance->createDebugUtilsMessengerEXTUnique(
        vk::DebugUtilsMessengerCreateInfoEXT({}, severityFlags, messageTypeFlags,
            &debugUtilsMessengerCallback));
}

void App::createSurface()
{
    VkSurfaceKHR _surface;
    if (glfwCreateWindowSurface(VkInstance(m_instance.get()), m_window, nullptr,
            &_surface)
        != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
    m_surface = vk::UniqueSurfaceKHR(_surface, { m_instance.get() });
}

void App::pickPhysicalDevice()
{
    for (const auto& device : m_instance->enumeratePhysicalDevices()) {
        if (isDeviceSuitable(device)) {
            m_physicalDevice = device;
            break;
        }
    }

    if (!m_physicalDevice) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

void App::createLogicalDevice()
{
    QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(),
        indices.presentFamily.value() };

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        vk::DeviceQueueCreateInfo queueCreateInfo({}, queueFamily, 1,
            &queuePriority);
        queueCreateInfos.push_back(queueCreateInfo);
    }

    vk::PhysicalDeviceFeatures deviceFeatures {};

    vk::DeviceCreateInfo createInfo({}, queueCreateInfos, {}, deviceExtensions,
        &deviceFeatures);

    if (enableValidationLayers) {
        createInfo.setPEnabledLayerNames(validationLayers);
    }

    m_device = m_physicalDevice.createDeviceUnique(createInfo);
    m_graphicsQueue = m_device->getQueue(indices.graphicsFamily.value(), 0);
    m_presentQueue = m_device->getQueue(indices.presentFamily.value(), 0);
}

void App::createSwapChain()
{
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(m_physicalDevice);

    vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR createInfo(
        {}, m_surface.get(), imageCount, surfaceFormat.format,
        surfaceFormat.colorSpace, extent, /* imageArrayLayers = */ 1,
        vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive, {},
        swapChainSupport.capabilities.currentTransform,
        vk::CompositeAlphaFlagBitsKHR::eOpaque, presentMode,
        /* clipped = */ VK_TRUE, nullptr);

    QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);
    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.setImageSharingMode(vk::SharingMode::eConcurrent);
        std::array familyIndices = { indices.graphicsFamily.value(),
            indices.presentFamily.value() };
        createInfo.setQueueFamilyIndices(familyIndices);
    }

    m_swapChain = m_device->createSwapchainKHRUnique(createInfo);
    m_swapChainImages = m_device->getSwapchainImagesKHR(m_swapChain.get());
    m_swapChainImageFormat = surfaceFormat.format;
    m_swapChainExtent = extent;
}

void App::createImageViews()
{
    m_swapChainImageViews.resize(m_swapChainImages.size());

    vk::ComponentMapping components(
        vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
        vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity);
    vk::ImageSubresourceRange subresourceRange(vk::ImageAspectFlagBits::eColor, 0,
        1, 0, 1);
    for (size_t i = 0; i < m_swapChainImages.size(); i++) {
        vk::ImageViewCreateInfo createInfo(
            {}, m_swapChainImages[i], vk::ImageViewType::e2D, m_swapChainImageFormat,
            components, subresourceRange);
        m_swapChainImageViews[i] = m_device->createImageViewUnique(createInfo);
    }
}

void App::createRenderPass()
{
    vk::AttachmentDescription colorAttachment(
        {}, m_swapChainImageFormat, vk::SampleCountFlagBits::e1,
        vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
        vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
        vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR);

    vk::AttachmentReference colorAttachmentRef(
        0, vk::ImageLayout::eColorAttachmentOptimal);

    vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, {}, {},
        1, &colorAttachmentRef);

    vk::SubpassDependency dependency(
        VK_SUBPASS_EXTERNAL, 0, vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eColorAttachmentOutput, {},
        vk::AccessFlagBits::eColorAttachmentWrite);

    vk::RenderPassCreateInfo renderPassInfo({}, colorAttachment, subpass,
        dependency);

    m_renderPass = m_device->createRenderPassUnique(renderPassInfo);
}

void App::createGraphicsPipeline()
{
    vk::UniqueShaderModule vertShaderModule = createShaderModule(shaderc_glsl_default_vertex_shader, R"vertexShader(
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(location = 0) out vec3 fragColor;

            vec2 positions[3] = vec2[](
                vec2(0.0, -0.5),
                vec2(0.5, 0.5),
                vec2(-0.5, 0.5)
            );

            vec3 colors[3] = vec3[](
                vec3(1.0, 0.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                vec3(0.0, 0.0, 1.0)
            );

            void main() {
                gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
                fragColor = colors[gl_VertexIndex];
            }
            )vertexShader");
    vk::UniqueShaderModule fragShaderModule = createShaderModule(
        shaderc_glsl_default_fragment_shader, R"fragmentShader(
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(location = 0) in vec3 fragColor;

            layout(location = 0) out vec4 outColor;

            void main() {
            outColor = vec4(fragColor, 1.0);
            }
            )fragmentShader");

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo(
        {}, vk::ShaderStageFlagBits::eVertex, vertShaderModule.get(), "main");
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo(
        {}, vk::ShaderStageFlagBits::eFragment, fragShaderModule.get(), "main");

    std::array shaderStages = { vertShaderStageInfo, fragShaderStageInfo };

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo({}, 0, nullptr, 0,
        nullptr);

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly(
        {}, vk::PrimitiveTopology::eTriangleList);

    vk::Viewport viewport(0.0f, 0.0f, (float)m_swapChainExtent.width,
        (float)m_swapChainExtent.height, 0.0f, 1.0f);

    vk::Rect2D scissor({ 0, 0 }, m_swapChainExtent);

    vk::PipelineViewportStateCreateInfo viewportState({}, 1, &viewport, 1,
        &scissor);

    vk::PipelineRasterizationStateCreateInfo rasterizer(
        {}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill,
        vk::CullModeFlagBits::eBack, vk::FrontFace::eClockwise, VK_FALSE, 0.0f,
        0.0f, 0.0f, 1.0f);

    vk::PipelineMultisampleStateCreateInfo multisampling(
        {}, vk::SampleCountFlagBits::e1, VK_FALSE);

    vk::PipelineColorBlendAttachmentState colorBlendAttachment(VK_FALSE);
    colorBlendAttachment.setColorWriteMask(
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

    vk::PipelineColorBlendStateCreateInfo colorBlending(
        {}, VK_FALSE, vk::LogicOp::eCopy, 1, &colorBlendAttachment);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo {};

    m_pipelineLayout = m_device->createPipelineLayoutUnique(pipelineLayoutInfo);

    vk::GraphicsPipelineCreateInfo pipelineInfo(
        {}, shaderStages, &vertexInputInfo, &inputAssembly, {}, &viewportState,
        &rasterizer, &multisampling, {}, &colorBlending, {}, m_pipelineLayout.get(),
        m_renderPass.get(), 0, {}, {});

    vk::ResultValue<vk::UniquePipeline> result = m_device->createGraphicsPipelineUnique({}, pipelineInfo);
    if (result.result == vk::Result::eSuccess) {
        m_graphicsPipeline = std::move(result.value);
    } else {
        throw std::runtime_error("failed to create a pipeline!");
    }
}

void App::createFramebuffers()
{
    m_swapChainFramebuffers.reserve(m_swapChainImageViews.size());

    for (auto const& view : m_swapChainImageViews) {
        vk::FramebufferCreateInfo framebufferInfo({}, m_renderPass.get(), view.get(),
            m_swapChainExtent.width,
            m_swapChainExtent.height, 1);
        m_swapChainFramebuffers.push_back(
            m_device->createFramebufferUnique(framebufferInfo));
    }
}

void App::createCommandPool()
{
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(m_physicalDevice);

    vk::CommandPoolCreateInfo poolInfo({},
        queueFamilyIndices.graphicsFamily.value());

    m_commandPool = m_device->createCommandPoolUnique(poolInfo);
}

void App::createCommandBuffers()
{
    vk::CommandBufferAllocateInfo allocInfo(
        m_commandPool.get(), vk::CommandBufferLevel::ePrimary,
        static_cast<uint32_t>(m_swapChainFramebuffers.size()));

    m_commandBuffers = m_device->allocateCommandBuffersUnique(allocInfo);

    for (size_t i = 0; i < m_commandBuffers.size(); i++) {
        m_commandBuffers[i]->begin(vk::CommandBufferBeginInfo {});

        vk::ClearValue clearColor(std::array { 0.0f, 0.0f, 0.0f, 1.0f });
        vk::RenderPassBeginInfo renderPassInfo(
            m_renderPass.get(), m_swapChainFramebuffers[i].get(),
            { { 0, 0 }, m_swapChainExtent }, clearColor);

        m_commandBuffers[i]->beginRenderPass(renderPassInfo,
            vk::SubpassContents::eInline);
        m_commandBuffers[i]->bindPipeline(vk::PipelineBindPoint::eGraphics,
            m_graphicsPipeline.get());
        m_commandBuffers[i]->draw(3, 1, 0, 0);
        m_commandBuffers[i]->endRenderPass();
        m_commandBuffers[i]->end();
    }
}

void App::createSyncObjects()
{
    m_imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    m_renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    m_inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    m_imagesInFlight.resize(m_swapChainImages.size());

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        m_imageAvailableSemaphores[i] = m_device->createSemaphoreUnique({});
        m_renderFinishedSemaphores[i] = m_device->createSemaphoreUnique({});
        m_inFlightFences[i] = m_device->createFence({ vk::FenceCreateFlagBits::eSignaled });
    }
}

void App::drawFrame()
{
    m_device->waitForFences(m_inFlightFences[m_currentFrame], true, UINT64_MAX);
    m_device->resetFences(m_inFlightFences[m_currentFrame]);

    vk::ResultValue<uint32_t> result = m_device->acquireNextImageKHR(m_swapChain.get(), UINT64_MAX,
        m_imageAvailableSemaphores[m_currentFrame].get());
    uint32_t imageIndex;
    if (result.result == vk::Result::eSuccess) {
        imageIndex = result.value;
    } else {
        throw std::runtime_error("failed to acquire next image!");
    }

    if (m_imagesInFlight[imageIndex]) {
        m_device->waitForFences(m_imagesInFlight[imageIndex], true, UINT64_MAX);
    }
    m_imagesInFlight[imageIndex] = m_inFlightFences[m_currentFrame];

    vk::PipelineStageFlags waitStage(
        vk::PipelineStageFlagBits::eColorAttachmentOutput);
    vk::SubmitInfo submitInfo(m_imageAvailableSemaphores[m_currentFrame].get(),
        waitStage, m_commandBuffers[imageIndex].get(),
        m_renderFinishedSemaphores[m_currentFrame].get());

    m_device->resetFences(m_inFlightFences[m_currentFrame]);

    m_graphicsQueue.submit(submitInfo, m_inFlightFences[m_currentFrame]);

    vk::PresentInfoKHR presentInfo(m_renderFinishedSemaphores[m_currentFrame].get(),
        m_swapChain.get(), imageIndex);

    m_graphicsQueue.presentKHR(presentInfo);

    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

vk::UniqueShaderModule App::createShaderModule(shaderc_shader_kind kind,
    const std::string& shader)
{
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetOptimizationLevel(shaderc_optimization_level_performance);

    shaderc::SpvCompilationResult spv = compiler.CompileGlslToSpv(shader, kind, "shader", options);
    if (spv.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::cerr << spv.GetErrorMessage();
    }
    auto shaderCode = std::vector<uint32_t> { spv.cbegin(), spv.cend() };
    auto size = std::distance(shaderCode.begin(), shaderCode.end());
    auto shaderCreateInfo = vk::ShaderModuleCreateInfo {
        {}, size * sizeof(uint32_t), shaderCode.data()
    };
    auto shaderModule = m_device->createShaderModuleUnique(shaderCreateInfo);

    return shaderModule;
}

vk::SurfaceFormatKHR App::chooseSwapSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

vk::PresentModeKHR App::chooseSwapPresentMode(
    const std::vector<vk::PresentModeKHR>& availablePresentModes)
{
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eFifoRelaxed) {
            return availablePresentMode;
        }
    }

    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D App::chooseSwapExtent(
    const vk::SurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    } else {
        int width, height;
        glfwGetFramebufferSize(m_window, &width, &height);

        vk::Extent2D actualExtent = { static_cast<uint32_t>(width),
            static_cast<uint32_t>(height) };

        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height);

        return actualExtent;
    }
}

SwapChainSupportDetails App::querySwapChainSupport(vk::PhysicalDevice device)
{
    SwapChainSupportDetails details;
    details.capabilities = device.getSurfaceCapabilitiesKHR(m_surface.get());
    details.formats = device.getSurfaceFormatsKHR(m_surface.get());
    details.presentModes = device.getSurfacePresentModesKHR(m_surface.get());

    return details;
}

bool App::isDeviceSuitable(vk::PhysicalDevice device)
{
    QueueFamilyIndices indices = findQueueFamilies(device);

    bool extensionsSupported = checkDeviceExtensionSupport(device);

    bool swapChainAdequate = false;
    if (extensionsSupported) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    return indices.isComplete() && extensionsSupported && swapChainAdequate;
}

bool App::checkDeviceExtensionSupport(vk::PhysicalDevice device)
{
    std::vector<vk::ExtensionProperties> availableExtensions = device.enumerateDeviceExtensionProperties();

    std::set<std::string> requiredExtensions(deviceExtensions.begin(),
        deviceExtensions.end());

    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

QueueFamilyIndices App::findQueueFamilies(vk::PhysicalDevice device)
{
    QueueFamilyIndices indices;

    std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();

    uint32_t i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
            indices.graphicsFamily = i;
        }

        if (device.getSurfaceSupportKHR(i, m_surface.get())) {
            indices.presentFamily = i;
        }

        if (indices.isComplete()) {
            break;
        }

        i++;
    }

    return indices;
}

std::vector<const char*> App::getRequiredExtensions()
{
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions,
        glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

bool App::checkValidationLayerSupport()
{
    std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();

    for (const char* layerName : validationLayers) {
        bool layerFound = false;
        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }
        if (!layerFound) {
            return false;
        }
    }
    return true;
}
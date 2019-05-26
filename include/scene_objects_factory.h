#pragma once

#include <scene_objects.h>

class SceneObjectsFactory {
public:
    using Ptr = std::shared_ptr<SceneObject>;

    template<typename T, typename ...Args>
    static Ptr create(Args &&...args) {
        return std::make_shared<T>(std::forward<Args>(args)...);
    }
};
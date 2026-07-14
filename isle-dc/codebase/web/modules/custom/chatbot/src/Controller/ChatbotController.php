<?php

namespace Drupal\chatbot\Controller;

use Drupal\Core\Controller\ControllerBase;
use Symfony\Component\HttpFoundation\JsonResponse;
use Symfony\Component\HttpFoundation\Request;

class ChatbotController extends ControllerBase {
public function page() {
  return [
    '#theme' => 'chatbot_page',
    '#attached' => [
      'library' => [
        'chatbot/chatbot',
      ],
    ],
  ];
}
public function sendMessage(Request $request) {

  $data = json_decode($request->getContent(), TRUE);

  $message = $data['message'] ?? '';

  return new JsonResponse([
    'reply' => 'You said: ' . $message,
  ]);

}

}